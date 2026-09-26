r"""The assembled front end: shapes, parameter arithmetic, guards and the normaliser ban.

Two things are checked here that no test of the parts could catch.

The first is **wiring**. Each piece is proven in its own file -- the decimator's offset, the
featurisation's gap handling -- but a stage that projected to the wrong width, or a cascade whose
strides did not multiply to the loader's decimation, would still be built from correct parts. The
parameter budget is therefore asserted against the arithmetic that produces it, per stage, rather
than against a single literal: a literal tells you a number changed, the arithmetic tells you which
term did.

The second is the **normaliser ban**. ``nn.GroupNorm`` reduces over $(C/G, T)$ within a group, so
one of them anywhere in this stack makes every history state carry an image of its own future. The
ban runs at construction, and the test that proves it plants a ``GroupNorm`` inside the convolution
block and requires the constructor to refuse.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.blocks import GatedCausalConvBlock
from teb_vae.lag_attn_transformer_e2e.nets import frontend as frontend_module
from teb_vae.lag_attn_transformer_e2e.nets.frontend import (
    FEATURE_CHANNELS,
    FRONTEND_KERNELS,
    NUM_STAGES,
)
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    build_frontend,
)


def _stage_parameters(in_channels: int, out_channels: int, kernel: int) -> int:
    r"""Parameters of one stage, from the arithmetic rather than from a measurement.

    $$
    \underbrace{C_{\mathrm{in}} C_{\mathrm{out}} + C_{\mathrm{out}}}_{\text{projection, with bias}}
    \;+\;
    \underbrace{3 C_{\mathrm{out}}^2 + 3 C_{\mathrm{out}} + C_{\mathrm{out}} k}_{
        \text{gated causal convolution block}} .
    $$

    The block's term is its own documented count: $2d^2$ for the gated input projection, $d^2$ for
    the output projection, $d$ apiece for two norms and the LayerScale, and $dk$ for the depthwise
    filter bank. The decimator contributes nothing, which is the point of it being a buffer.

    Args:
        in_channels: Stage input width.
        out_channels: Stage output width.
        kernel: Depthwise kernel width.

    Returns:
        The parameter count.
    """
    projection = in_channels * out_channels + out_channels
    block = 3 * out_channels**2 + 3 * out_channels + out_channels * kernel
    return projection + block


def _widths(d_model: int) -> tuple:
    """The stage output widths the front end derives from ``d_model``."""
    quarter = d_model // 4
    return (quarter, 2 * quarter, 3 * quarter, d_model)


# ---------------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------------
def test_the_production_geometry_maps_raw_onto_the_token_grid():
    """$(B, rT)$ raw and $(B, T)$ weight in, $(B, T, d_{\\mathrm{model}})$ out -- the shape the
    encoder that replaces the stored feature adapters expects, unchanged."""
    steps = int(SHIPPED_KWARGS["sequence_length"])
    raw_per_step = int(SHIPPED_KWARGS["raw_per_step"])
    net = build_frontend(SHIPPED_KWARGS)

    with torch.no_grad():
        out = net(torch.randn(2, steps * raw_per_step), torch.ones(2, steps))

    assert out.shape == (2, steps, int(SHIPPED_KWARGS["d_model"]))


def test_a_fully_invalid_batch_still_produces_a_finite_non_zero_token():
    """The featurisation emits an exactly zero vector for a fully masked window, and an exactly zero
    token entering repeated pre-normalisation is the accident the sibling's input adapter documents
    reaching gradient norms around $10^{26}$. The stage projections carry a bias for that reason.

    Measured on **both of the model's** front ends, not on a standalone one: a front end built
    directly has torch's own non-zero ``nn.Linear`` bias and passes this on any code, while the
    model runs ``initialization``, which zeros every ``nn.Linear`` bias it walks -- so only the
    model's own front ends can say whether the bias was restored on the objects that train.
    """
    model = SeqVaeLagAttnTrfE2E(**TINY_KWARGS).eval()
    raw_per_step = int(TINY_KWARGS["raw_per_step"])
    raw = torch.randn(2, SEQ_LEN * raw_per_step)
    dead = torch.zeros(2, SEQ_LEN)

    for frontend in (model.target_frontend, model.source_frontend):
        with torch.no_grad():
            out = frontend(raw, dead)

        assert bool(torch.isfinite(out).all())
        assert float(out.abs().max()) > 0.0, (
            "a fully invalid window emits an exactly zero token from the model's own front end: "
            "the stage projections' biases were zeroed and never restored"
        )


# ---------------------------------------------------------------------------------------
# The parameter budget
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_parameter_count_is_the_per_stage_arithmetic(kwargs):
    net = build_frontend(kwargs)
    d_model = int(kwargs["d_model"])
    widths = _widths(d_model)
    kernels = tuple(kwargs.get("frontend_kernels", FRONTEND_KERNELS))

    subtotals = [
        _stage_parameters(in_channels, out_channels, kernel)
        for in_channels, out_channels, kernel in zip(
            (FEATURE_CHANNELS,) + widths[:-1], widths, kernels
        )
    ]
    for stage, expected in zip(net.stage_modules, subtotals):
        assert sum(parameter.numel() for parameter in stage.parameters()) == expected

    # The trailing RMSNorm is the only parameter outside the stages.
    expected_total = sum(subtotals) + d_model
    assert sum(parameter.numel() for parameter in net.parameters()) == expected_total


# ---------------------------------------------------------------------------------------
# Construction and forward guards
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"d_model": 30}, "divisible by 4"),
        ({"kernels": (3, 3, 3)}, f"expected {NUM_STAGES} kernels"),
        # The decimation convention and the anchor convention have to agree; a disagreement
        # produces a correctly-shaped tensor on the wrong grid.
        ({"raw_per_step": 8}, "disagrees with the front end's total stride"),
    ],
    ids=["indivisible-width", "wrong-kernel-count", "stride-mismatch"],
)
def test_an_inconsistent_front_end_is_refused_by_name(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_frontend(TINY_KWARGS, **overrides)


def test_a_raw_signal_of_the_wrong_length_is_refused_by_name():
    net = build_frontend(TINY_KWARGS)

    with pytest.raises(ValueError, match="expected a raw signal of"):
        net(torch.randn(2, 8 * SEQ_LEN), torch.ones(2, SEQ_LEN))


# ---------------------------------------------------------------------------------------
# The normaliser ban
# ---------------------------------------------------------------------------------------
def test_a_planted_group_norm_is_refused_at_construction(monkeypatch):
    """The negative control for the whole ban, planted where a real edit would put one: inside the
    convolution block, reached only through the stage. If the walk ran at test time rather than at
    construction, this would build a leaking front end and report nothing.
    """

    class _LeakyBlock(GatedCausalConvBlock):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.leak = nn.GroupNorm(1, self.d_model)

    monkeypatch.setattr(frontend_module, "GatedCausalConvBlock", _LeakyBlock)

    with pytest.raises(ValueError, match="GroupNorm"):
        build_frontend(TINY_KWARGS)
