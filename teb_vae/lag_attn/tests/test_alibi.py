r"""The lag-decay bias: a learnable ``(num_heads, L)`` addition to the attention scores.

With ``lag_bias_init='alibi_decay'`` each head starts with $b_{h\ell} = -m_h \ell$ added to its
scores, so long lags begin penalised and the model must earn a long-lag reading. The bias is a
*parameter*, not a fixed prior -- a head that finds real long-lag structure can flatten its own
decay. What it cannot do is start there by accident.

These tests pin that the bias is seeded correctly, scales with ``alibi_slope_scale``, trains, and is
absent under ``'normal'``; and, for the attention itself, the causal lag mask (a lag $\ell$ at step
$t$ exists only when $t - \ell \ge 0$), row normalisation under softmax and entmax, the forward
shapes, and that gradient checkpointing leaves the output unchanged.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.attention import LagCrossAttention, alibi_slopes

_D_MODEL, _NUM_HEADS, _D_HEAD, _MAX_LAG = 32, 4, 8, 8
_BATCH, _SEQ_LEN = 2, 16


def _make(**overrides) -> LagCrossAttention:
    torch.manual_seed(0)
    kwargs = dict(
        d_model=_D_MODEL,
        num_heads=_NUM_HEADS,
        d_head=_D_HEAD,
        max_lag=_MAX_LAG,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return LagCrossAttention(**kwargs)


def _states():
    generator = torch.Generator().manual_seed(0)
    h_y = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=generator)
    h_u = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=generator)
    return h_y, h_u


def test_alibi_bias_is_seeded_with_a_negative_lag_slope():
    r"""Each head's bias must decay monotonically with lag: $b_{h\ell} = -m_h \ell$."""
    bias = _make(lag_bias_init="alibi_decay").lag_score_bias.detach()

    assert torch.allclose(bias[:, 0], torch.zeros(_NUM_HEADS))
    # Strictly decreasing along the lag axis for every head.
    assert bool((bias[:, 1:] < bias[:, :-1]).all())
    # Distinct per-head slopes: the heads must span scales, not duplicate one.
    slopes = bias[:, 0] - bias[:, 1]
    assert len(torch.unique(slopes)) == _NUM_HEADS


def test_normal_lag_bias_registers_no_parameter():
    attention = _make(lag_bias_init="normal")
    assert attention.lag_score_bias is None
    assert "lag_score_bias" not in dict(attention.named_parameters())


def test_alibi_bias_receives_gradient():
    attention = _make(lag_bias_init="alibi_decay")
    assert attention.lag_score_bias.shape == (_NUM_HEADS, _MAX_LAG + 1)
    # Registered, so the optimiser and DDP both see it.
    assert "lag_score_bias" in dict(attention.named_parameters())
    out, _, _ = attention(*_states())
    out.sum().backward()

    grad = attention.lag_score_bias.grad
    assert grad is not None, "lag_score_bias got no gradient"
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0.0


def test_invalid_lag_bias_init_raises():
    with pytest.raises(ValueError, match="lag_bias_init"):
        _make(lag_bias_init="exponential")


def test_alibi_slope_scale_softens_the_slope():
    base = _make(lag_bias_init="alibi_decay")
    soft = _make(lag_bias_init="alibi_decay", alibi_slope_scale=0.25)

    assert torch.allclose(
        soft.lag_score_bias.detach(), 0.25 * base.lag_score_bias.detach()
    )
    # The softened bias penalises the far lag ~4x less, so distant lags stay reachable.
    assert bool(
        (
            soft.lag_score_bias.detach()[:, _MAX_LAG]
            > base.lag_score_bias.detach()[:, _MAX_LAG]
        ).all()
    )


def test_alibi_slopes_follow_the_power_of_two_schedule():
    r"""Press et al. (2022): for $n = 8$ heads the slopes are $m_h = 2^{-h}$, $h = 1, \dots, 8$."""
    slopes = alibi_slopes(8)
    expected = 2.0 ** -torch.arange(1, 9, dtype=slopes.dtype)
    assert torch.allclose(slopes, expected)


def test_build_lag_mask_is_public_and_lower_triangular():
    """The model composes this with its own band mask, so it is part of the interface."""
    attention = _make()
    mask = attention.build_lag_mask(_SEQ_LEN)

    assert mask.shape == (_SEQ_LEN, _MAX_LAG + 1)
    assert mask.dtype == torch.bool
    # Lag l at step t refers to step t-l, which exists only when t >= l.
    for step in range(_SEQ_LEN):
        for lag in range(_MAX_LAG + 1):
            assert bool(mask[step, lag]) == (step - lag >= 0)


def test_masked_lags_receive_exactly_zero_attention():
    """At step 0 only lag 0 exists; every other lag must get exactly zero, not merely little."""
    attention = _make().eval()
    with torch.no_grad():
        _, alpha, _ = attention(*_states())

    assert torch.all(alpha[:, 0, :, 1:] == 0.0)


def test_entmax_and_softmax_rows_both_normalise():
    for use_entmax in (False, True):
        attention = _make(use_entmax=use_entmax).eval()
        with torch.no_grad():
            _, alpha, _ = attention(*_states())
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_entmax_is_sparser_than_softmax():
    """The reason entmax is a hard dependency rather than a nicety.

    Softmax cannot output an exact zero, so every one of the $L$ lags always carries some weight.
    Read as "which lag mattered", that is a smear; entmax can say *none*.
    """
    dense = _make(use_entmax=False).eval()
    sparse = _make(use_entmax=True).eval()
    with torch.no_grad():
        _, dense_alpha, _ = dense(*_states())
        _, sparse_alpha, _ = sparse(*_states())

    # Count exact zeros away from the structurally-masked early steps.
    dense_zeros = (dense_alpha[:, -1] == 0.0).sum().item()
    sparse_zeros = (sparse_alpha[:, -1] == 0.0).sum().item()
    assert dense_zeros == 0
    assert sparse_zeros > 0


def test_forward_shapes():
    attention = _make().eval()
    h_y, h_u = _states()
    with torch.no_grad():
        out, alpha, head_out = attention(h_y, h_u)

    assert out.shape == (_BATCH, _SEQ_LEN, _D_MODEL)
    assert alpha.shape == (_BATCH, _SEQ_LEN, _NUM_HEADS, _MAX_LAG + 1)
    assert head_out.shape == (_BATCH, _SEQ_LEN, _NUM_HEADS, _D_HEAD)


def test_a_batched_mask_of_identical_rows_equals_the_shared_mask():
    """A ``(B, T, L)`` mask is applied per sample; with every row the shared mask it is that mask."""
    attention = _make().eval()
    h_y, h_u = _states()
    flat = attention.build_lag_mask(_SEQ_LEN)
    batched = flat[None].expand(_BATCH, -1, -1)

    with torch.no_grad():
        from_flat = attention(h_y, h_u, flat)[1]
        from_batched = attention(h_y, h_u, batched)[1]

    assert torch.equal(from_flat, from_batched)


def test_a_per_sample_mask_moves_only_its_own_sample():
    """A key removed from sample 1 gets zero weight there, and sample 0 attends exactly as before.

    This is how the source-quality mask removes a bad source step: it must not leak into the other
    samples of the batch, which the earlier collapse to the first row silently did.
    """
    attention = _make().eval()
    h_y, h_u = _states()
    flat = attention.build_lag_mask(_SEQ_LEN)
    batched = flat[None].expand(_BATCH, -1, -1).clone()
    removed_step = _SEQ_LEN // 2
    for t in range(removed_step, min(_SEQ_LEN, removed_step + _MAX_LAG + 1)):
        batched[1, t, t - removed_step] = False

    with torch.no_grad():
        shared = attention(h_y, h_u, flat)[1]
        masked = attention(h_y, h_u, batched)[1]

    assert torch.equal(masked[0], shared[0])
    for t in range(removed_step, min(_SEQ_LEN, removed_step + _MAX_LAG + 1)):
        assert float(masked[1, t, :, t - removed_step].abs().max()) == 0.0
    assert not torch.equal(masked[1], shared[1])


@pytest.mark.parametrize("use_entmax", [True, False])
def test_a_row_with_no_valid_key_attends_to_nothing_and_stays_finite(use_entmax):
    """All keys removed: zero weights and a zero summary, finite gradients, under both normalisers.

    ``entmax15`` raises an index error on an all ``-inf`` row, so the attention scores such a row
    as zeros and zeroes its weights afterwards instead.
    """
    attention = _make(use_entmax=use_entmax).train()
    h_y, h_u = _states()
    h_u = h_u.clone().requires_grad_(True)
    mask = attention.build_lag_mask(_SEQ_LEN)[None].expand(_BATCH, -1, -1).clone()
    mask[0] = False

    _, alpha, heads = attention(h_y, h_u, mask)
    assert float(alpha[0].abs().max()) == 0.0
    assert float(heads[0].abs().max()) == 0.0
    heads.sum().backward()
    assert torch.isfinite(h_u.grad).all()


def test_no_mask_is_identical_to_the_built_mask():
    attention = _make().eval()
    h_y, h_u = _states()
    with torch.no_grad():
        implicit = attention(h_y, h_u, None)[1]
        explicit = attention(h_y, h_u, attention.build_lag_mask(_SEQ_LEN))[1]
    assert torch.equal(implicit, explicit)


def test_gradient_checkpointing_matches_the_plain_path():
    """Checkpointing trades compute for memory; it must not trade away the answer."""
    h_y, h_u = _states()

    plain = _make(grad_checkpoint=False).train()
    checkpointed = _make(grad_checkpoint=True).train()
    checkpointed.load_state_dict(plain.state_dict())

    plain_out, _, _ = plain(h_y, h_u)
    checkpointed_out, _, _ = checkpointed(h_y, h_u)

    assert torch.allclose(plain_out, checkpointed_out, atol=1e-6)
