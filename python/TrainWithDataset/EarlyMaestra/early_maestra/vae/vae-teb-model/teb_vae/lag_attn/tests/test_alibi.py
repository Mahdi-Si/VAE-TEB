r"""The lag-decay bias: a learnable ``(num_heads, L)`` addition to the attention scores.

With ``lag_bias_init='alibi_decay'`` each head starts with $b_{h\ell} = -m_h \ell$ added to its
scores, so long lags begin penalised and the model must earn a long-lag reading. The bias is a
*parameter*, not a fixed prior -- a head that finds real long-lag structure can flatten its own
decay. What it cannot do is start there by accident.

These tests pin that the bias exists, is shaped and seeded correctly, trains, and is absent under
``'normal'``.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

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


def test_alibi_bias_is_a_trainable_parameter():
    attention = _make(lag_bias_init="alibi_decay")
    bias = attention.lag_score_bias

    assert isinstance(bias, nn.Parameter)
    assert bias.shape == (_NUM_HEADS, _MAX_LAG + 1)
    assert bias.requires_grad
    # Registered, so the optimiser and DDP both see it.
    assert "lag_score_bias" in dict(attention.named_parameters())


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
    out, _, _ = attention(*_states())
    out.sum().backward()

    grad = attention.lag_score_bias.grad
    assert grad is not None, "lag_score_bias got no gradient"
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0.0


def test_invalid_lag_bias_init_raises():
    with pytest.raises(ValueError, match="lag_bias_init"):
        _make(lag_bias_init="exponential")


def test_head_width_must_tile_the_model_width():
    with pytest.raises(ValueError, match="must equal d_model"):
        LagCrossAttention(d_model=32, num_heads=4, d_head=9, max_lag=_MAX_LAG)


def test_alibi_slope_scale_default_is_identity():
    default = _make(lag_bias_init="alibi_decay")
    explicit = _make(lag_bias_init="alibi_decay", alibi_slope_scale=1.0)
    assert torch.equal(
        default.lag_score_bias.detach(), explicit.lag_score_bias.detach()
    )


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


def test_alibi_slope_scale_zero_gives_a_flat_but_learnable_bias():
    attention = _make(lag_bias_init="alibi_decay", alibi_slope_scale=0.0)
    assert torch.allclose(attention.lag_score_bias.detach(), torch.zeros(1))
    assert attention.lag_score_bias.requires_grad


def test_alibi_slopes_follow_the_power_of_two_schedule():
    slopes = alibi_slopes(8)
    assert slopes.shape == (8,)
    assert bool((slopes > 0).all())
    # Geometric and decreasing: consecutive ratios are equal.
    ratios = slopes[1:] / slopes[:-1]
    assert torch.allclose(ratios, ratios[0].expand_as(ratios))


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


def test_a_batched_mask_is_accepted_and_collapsed():
    """A ``(B, T, L)`` mask can only depend on ``(t, l)``, so it collapses to the first row."""
    attention = _make().eval()
    h_y, h_u = _states()
    flat = attention.build_lag_mask(_SEQ_LEN)
    batched = flat[None].expand(_BATCH, -1, -1)

    with torch.no_grad():
        from_flat = attention(h_y, h_u, flat)[1]
        from_batched = attention(h_y, h_u, batched)[1]

    assert torch.equal(from_flat, from_batched)


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
