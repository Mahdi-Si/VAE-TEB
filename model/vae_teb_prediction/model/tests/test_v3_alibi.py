r"""S3-T01: ALiBi lag decay is a learnable ``(num_heads, L)`` attention-score bias.

``lag_bias_init='alibi_decay'`` (G5) is a pure configuration flip -- v1's
:class:`LagCrossAttention` already builds the parameter. These tests pin the two properties
Sprint 3 depends on: the bias exists and trains under ``'alibi_decay'``, and it is absent
under ``'normal'`` so the Sprint-0 golden parity against v1 is untouched.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


def test_alibi_bias_is_a_trainable_parameter(tiny_kwargs):
    model = SeqVaeLagAttnV3(lag_bias_init="alibi_decay", **tiny_kwargs)
    bias = model.lag_attn.lag_score_bias

    assert isinstance(bias, nn.Parameter)
    assert bias.shape == (tiny_kwargs["num_heads"], tiny_kwargs["max_lag"] + 1)
    assert bias.requires_grad
    # It is registered under the model, so the optimiser and DDP both see it.
    assert "lag_attn.lag_score_bias" in dict(model.named_parameters())


def test_alibi_bias_is_seeded_with_a_negative_lag_slope(tiny_kwargs):
    r"""Each head's bias must decay monotonically with lag: :math:`b_{h\ell} = -m_h \ell`."""
    model = SeqVaeLagAttnV3(lag_bias_init="alibi_decay", **tiny_kwargs)
    bias = model.lag_attn.lag_score_bias.detach()

    assert torch.allclose(bias[:, 0], torch.zeros(tiny_kwargs["num_heads"]))
    # Strictly decreasing along the lag axis for every head.
    assert bool((bias[:, 1:] < bias[:, :-1]).all())
    # Distinct per-head slopes (the ALiBi power-of-two schedule).
    slopes = bias[:, 0] - bias[:, 1]
    assert len(torch.unique(slopes)) == tiny_kwargs["num_heads"]


def test_normal_lag_bias_registers_no_parameter(tiny_kwargs):
    """The parity path must not add a parameter (v1 has none)."""
    model = SeqVaeLagAttnV3(lag_bias_init="normal", **tiny_kwargs)
    assert model.lag_attn.lag_score_bias is None
    assert "lag_attn.lag_score_bias" not in dict(model.named_parameters())


def test_alibi_bias_receives_gradient(tiny_kwargs, inputs):
    model = SeqVaeLagAttnV3(lag_bias_init="alibi_decay", **tiny_kwargs)
    outs = model(*inputs)
    model.compute_loss(outs, inputs[0], inputs[1], beta=0.1)["total_loss"].backward()

    grad = model.lag_attn.lag_score_bias.grad
    assert grad is not None, "lag_score_bias got no gradient"
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0.0


def test_invalid_lag_bias_init_raises(tiny_kwargs):
    with pytest.raises(ValueError):
        SeqVaeLagAttnV3(lag_bias_init="exponential", **tiny_kwargs)


def test_alibi_slope_scale_default_is_identity(tiny_kwargs):
    r"""``alibi_slope_scale=1.0`` (default) reproduces the standard ALiBi init exactly."""
    default = SeqVaeLagAttnV3(lag_bias_init="alibi_decay", **tiny_kwargs)
    explicit = SeqVaeLagAttnV3(
        lag_bias_init="alibi_decay", alibi_slope_scale=1.0, **tiny_kwargs
    )
    assert torch.equal(
        default.lag_attn.lag_score_bias.detach(),
        explicit.lag_attn.lag_score_bias.detach(),
    )


def test_alibi_slope_scale_softens_the_slope(tiny_kwargs):
    r"""``alibi_slope_scale=s`` scales the per-lag decay bias by ``s`` (softer long-lag prior)."""
    base = SeqVaeLagAttnV3(lag_bias_init="alibi_decay", **tiny_kwargs)
    soft = SeqVaeLagAttnV3(
        lag_bias_init="alibi_decay", alibi_slope_scale=0.25, **tiny_kwargs
    )
    assert torch.allclose(
        soft.lag_attn.lag_score_bias.detach(),
        0.25 * base.lag_attn.lag_score_bias.detach(),
    )
    # The softened bias penalises the far lag ~4x less, so distant lags stay reachable.
    far = tiny_kwargs["max_lag"]
    assert bool(
        (soft.lag_attn.lag_score_bias.detach()[:, far]
         > base.lag_attn.lag_score_bias.detach()[:, far]).all()
    )
