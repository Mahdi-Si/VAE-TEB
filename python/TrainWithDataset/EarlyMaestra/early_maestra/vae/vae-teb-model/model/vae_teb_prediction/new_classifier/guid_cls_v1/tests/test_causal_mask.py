"""Causal-mask correctness tests for ``RelativeTimeMultiHeadSelfAttention``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.temporal_transformer import (  # noqa: E402
    RelativeTimeMultiHeadSelfAttention,
    RelativeTimeTransformer,
)


def _build_inputs(B: int, N: int, d_model: int):
    x = torch.randn(B, N, d_model, requires_grad=True)
    seg_mask = torch.ones(B, N, dtype=torch.bool)
    rel_idx = torch.zeros(B, N, N, dtype=torch.long)
    return x, seg_mask, rel_idx


def test_causal_mask_blocks_future_attention() -> None:
    """Position ``n`` must not attend to positions ``j > n`` when causal=True.

    Concretely, the gradient of ``y[:, n].sum()`` w.r.t. ``x[:, n+1:, :]``
    must be exactly zero. Negative control with ``causal=False`` shows the
    same gradient is non-zero.
    """
    B, N, d_model = 1, 6, 32
    n_heads, d_head = 4, 8
    attn_causal = RelativeTimeMultiHeadSelfAttention(
        d_model=d_model, n_heads=n_heads, d_head=d_head, n_buckets=4, causal=True
    )
    attn_bidir = RelativeTimeMultiHeadSelfAttention(
        d_model=d_model, n_heads=n_heads, d_head=d_head, n_buckets=4, causal=False
    )

    x, seg_mask, rel_idx = _build_inputs(B, N, d_model)

    # Causal: gradient at positions > n must be zero.
    y = attn_causal(x, seg_mask, rel_idx)
    for n in range(N - 1):
        x.grad = None
        y[:, n].sum().backward(retain_graph=True)
        assert x.grad is not None
        future_grad = x.grad[:, n + 1 :, :]
        assert torch.all(future_grad == 0), (
            f"causal=True leaked future at position {n}: "
            f"max future-grad |{future_grad.abs().max().item()}|"
        )

    # Bidirectional negative control: at least one future grad is non-zero.
    x2, seg_mask2, rel_idx2 = _build_inputs(B, N, d_model)
    y2 = attn_bidir(x2, seg_mask2, rel_idx2)
    y2[:, 0].sum().backward()
    assert x2.grad is not None
    assert x2.grad[:, 1:, :].abs().sum() > 0


def test_position_zero_finite_under_causal_mask() -> None:
    """Position 0 attends only to itself; output must remain finite."""
    B, N, d_model = 2, 5, 32
    attn = RelativeTimeMultiHeadSelfAttention(
        d_model=d_model, n_heads=4, d_head=8, n_buckets=4, causal=True
    )
    x, seg_mask, rel_idx = _build_inputs(B, N, d_model)
    y = attn(x, seg_mask, rel_idx)
    assert torch.all(torch.isfinite(y[:, 0]))


def test_full_transformer_remains_causal() -> None:
    """Stacked causal blocks preserve the no-future-leakage property."""
    B, N, d_model = 1, 5, 32
    transformer = RelativeTimeTransformer(
        d_model=d_model, n_heads=4, d_head=8, n_layers=2, n_buckets=4, causal=True
    )
    x, seg_mask, rel_idx = _build_inputs(B, N, d_model)
    y = transformer(x, seg_mask, rel_idx)

    for n in range(N - 1):
        x.grad = None
        y[:, n].sum().backward(retain_graph=True)
        grad = x.grad
        assert grad is not None
        assert torch.all(grad[:, n + 1 :, :] == 0)
