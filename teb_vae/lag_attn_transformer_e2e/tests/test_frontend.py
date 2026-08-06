r"""The assembled front end: its shapes, its parameter arithmetic, and the normalisers it refuses.

Two things are checked here that no test of the parts could catch.

The first is **wiring**. Each piece below is proven in its own file -- the decimator's offset, the
featurisation's gap handling -- but a stage that projected to the wrong width, or a cascade whose
strides did not multiply to $16$, would still be built from correct parts. The parameter budget is
therefore asserted against the arithmetic that produces it, per stage, rather than against a single
literal somebody read off a run: a literal tells you a number changed, the arithmetic tells you
which term did.

The second is the **normaliser ban**. ``nn.GroupNorm`` reduces over $(C/G, T)$ within a group, so
one of them anywhere in this stack makes every history state carry an image of its own future --
which is the entire property this input representation exists to buy, void, with no symptom in any
loss curve. The ban runs at construction, and the test that proves it plants a ``GroupNorm`` inside
the convolution block and requires the constructor to refuse.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    LAYER_SCALE_INIT,
    CausalDepthwiseConv1d,
    GatedCausalConvBlock,
    LayerScale,
    RMSNorm,
    init_depthwise_,
)
from teb_vae.lag_attn_transformer_e2e.nets import frontend as frontend_module
from teb_vae.lag_attn_transformer_e2e.nets.frontend import (
    ANTI_ALIAS_TAPS,
    FEATURE_CHANNELS,
    FRONTEND_KERNELS,
    NUM_STAGES,
    TIME_POOLING_NORMS,
    CausalAntiAliasDecimate,
    refuse_time_pooling_norms,
)
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    build_frontend,
    make_stub_batch,
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
    """$(B, 4800)$ raw and $(B, 300)$ weight in, $(B, 300, 128)$ out -- the shape the encoder that
    replaces the stored feature adapters expects, unchanged."""
    steps = int(SHIPPED_KWARGS["sequence_length"])
    raw_per_step = int(SHIPPED_KWARGS["raw_per_step"])
    net = build_frontend(SHIPPED_KWARGS)

    with torch.no_grad():
        out = net(torch.randn(2, steps * raw_per_step), torch.ones(2, steps))

    assert out.shape == (2, steps, int(SHIPPED_KWARGS["d_model"]))


def test_the_tiny_geometry_maps_its_own_shapes():
    batch = make_stub_batch(BATCH, SEQ_LEN)
    net = build_frontend(TINY_KWARGS)

    with torch.no_grad():
        target = net(batch.fhr, batch.weight)
        source = net(batch.up, batch.weight)

    assert target.shape == (BATCH, SEQ_LEN, int(TINY_KWARGS["d_model"]))
    assert source.shape == target.shape


def test_the_total_stride_is_the_loader_decimation():
    """Token $t$'s newest input sample is ``total_stride * (t + 1) - 1``. If the stride were
    anything but the loader's own decimation the two grids would be different sequences with the
    same length, which nothing downstream could detect."""
    net = build_frontend(TINY_KWARGS)

    assert net.total_stride == int(TINY_KWARGS["raw_per_step"])
    assert net.total_stride == 2**NUM_STAGES


def test_a_fully_invalid_batch_still_produces_finite_output():
    """The featurisation emits an exactly zero vector for a fully masked window, and an exactly zero
    token entering repeated pre-normalisation is the accident the sibling's input adapter documents
    reaching gradient norms around $10^{26}$. The stage projections carry a bias for that reason;
    this is the assertion that keeps the reason true.

    Measured on the **model's** front end, not on a standalone one. A front end built directly has
    torch's own non-zero ``nn.Linear`` bias and passes this on any code; the model runs
    ``initialization``, which zeros every ``nn.Linear`` bias it walks, so only the model's own front
    end can say whether the bias survived to the object that trains. Asserted here against a
    standalone build so the two are shown to agree rather than assumed to.
    """
    model = SeqVaeLagAttnTrfE2E(**TINY_KWARGS).eval()
    raw_per_step = int(TINY_KWARGS["raw_per_step"])
    raw = torch.randn(2, SEQ_LEN * raw_per_step)
    dead = torch.zeros(2, SEQ_LEN)

    with torch.no_grad():
        out = model.target_frontend(raw, dead)
        standalone = build_frontend(TINY_KWARGS)(raw, dead)

    assert bool(torch.isfinite(out).all())
    assert float(out.abs().max()) > 0.0, (
        "a fully invalid window emits an exactly zero token from the model's own front end: the "
        "stage projections' biases were zeroed and never restored"
    )
    assert float(standalone.abs().max()) > 0.0


def test_both_front_ends_of_the_model_keep_their_stage_bias():
    """The paired structural half of the test above, and the one that names the cause.

    Every other operator in the cascade is bias-free -- the convolution block's projections, its
    depthwise convolution, ``RMSNorm`` and ``LayerScale`` -- so the stage projection's bias is the
    only thing standing between a fully invalid window and an exactly zero token. It is also the
    only bias the model's generic initialisation pass has any reason to touch, which is what makes
    this worth asserting on the built model rather than trusting to the constructor.
    """
    model = SeqVaeLagAttnTrfE2E(**TINY_KWARGS)

    for frontend in (model.target_frontend, model.source_frontend):
        for index, stage in enumerate(frontend.stage_modules):
            assert stage.proj.bias is not None
            assert float(stage.proj.bias.abs().max()) > 0.0, (
                f"stage {index}'s projection bias is all zeros after initialisation"
            )


# ---------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------
def test_every_stage_is_projection_then_gated_convolution_then_decimation():
    net = build_frontend(SHIPPED_KWARGS)
    widths = _widths(int(SHIPPED_KWARGS["d_model"]))
    inputs = (FEATURE_CHANNELS,) + widths[:-1]

    assert len(net.stage_modules) == NUM_STAGES
    for stage, in_channels, out_channels, kernel in zip(
        net.stage_modules, inputs, widths, FRONTEND_KERNELS
    ):
        assert isinstance(stage.proj, nn.Linear)
        assert (stage.proj.in_features, stage.proj.out_features) == (in_channels, out_channels)
        assert isinstance(stage.block, GatedCausalConvBlock)
        assert stage.block.conv.kernel_size == kernel
        assert isinstance(stage.decimate, CausalAntiAliasDecimate)
        assert stage.decimate.stride == 2


def test_every_stage_holds_a_depthwise_convolution_the_repair_pass_can_find():
    """``init_depthwise_`` detects ``CausalDepthwiseConv1d`` by class. Building the stages from the
    sibling's block rather than from a local convolution is what makes the front end visible to it
    with no extension -- and without the repair the depthwise weights start $8.03\\times$ too quiet,
    independent of the kernel, which no shape test could see."""
    net = build_frontend(SHIPPED_KWARGS)

    for stage in net.stage_modules:
        assert isinstance(stage.block.conv, CausalDepthwiseConv1d)
    assert init_depthwise_(net) == NUM_STAGES


def test_there_is_exactly_one_fixed_filter_per_stage_and_none_of_them_is_saved():
    """One decimator per stage, so the residual add happens at full rate and a single operator runs
    on the sum -- the "keep the skip path sample-aligned" hazard does not exist here rather than
    being tested for. And no filter reaches the checkpoint: they are constants of the architecture,
    and a saved one would make a checkpoint fail to load the moment the tap count changed."""
    net = build_frontend(SHIPPED_KWARGS)

    decimators = [child for child in net.modules() if isinstance(child, CausalAntiAliasDecimate)]
    filters = [name for name, _ in net.named_buffers() if name.endswith("fir")]

    assert len(decimators) == NUM_STAGES
    assert len(filters) == NUM_STAGES
    assert all(child.taps == ANTI_ALIAS_TAPS for child in decimators)
    assert not [key for key in net.state_dict() if "fir" in key]


def test_the_stack_ends_in_a_channel_axis_normaliser():
    """Everything downstream is calibrated to a normalised state; a pre-norm residual stack without
    a final norm exports a stream whose scale grows with depth."""
    net = build_frontend(SHIPPED_KWARGS)

    assert isinstance(net.output_norm, RMSNorm)
    assert net.output_norm.dim == int(SHIPPED_KWARGS["d_model"])


def test_a_fresh_front_end_starts_close_to_its_linear_path():
    """Recorded rather than discovered: ``LayerScale`` at $10^{-2}$ makes every convolution block's
    residual branch a hundredth of its eventual weight, so a freshly initialised front end is
    approximately a linear mix of the decimated ``[value, mask, delta]`` channels. That is a sane
    start, and the first epochs are the stages finding temporal structure."""
    net = build_frontend(SHIPPED_KWARGS)

    scales = [child for child in net.modules() if isinstance(child, LayerScale)]
    assert scales
    for scale in scales:
        assert torch.equal(
            scale.weight.detach(), torch.full_like(scale.weight.detach(), LAYER_SCALE_INIT)
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


def test_the_two_production_front_ends_cost_less_than_a_quarter_million_parameters():
    """A bound rather than a literal: the absolute total is pinned once, against the shipped config,
    by the document test. What matters here is the order of magnitude the design was accepted at --
    the two stored-feature adapters this replaces cost $156{,}288$."""
    net = build_frontend(SHIPPED_KWARGS)

    per_stream = sum(parameter.numel() for parameter in net.parameters())

    assert 100_000 < per_stream < 125_000
    assert 2 * per_stream < 250_000


# ---------------------------------------------------------------------------------------
# Construction and forward guards
# ---------------------------------------------------------------------------------------
def test_an_indivisible_width_is_refused_by_name():
    with pytest.raises(ValueError, match="divisible by 4"):
        build_frontend(TINY_KWARGS, d_model=30)


def test_a_wrong_number_of_kernels_is_refused_by_name():
    with pytest.raises(ValueError, match=f"expected {NUM_STAGES} kernels"):
        build_frontend(TINY_KWARGS, kernels=(3, 3, 3))


def test_a_stride_that_disagrees_with_the_loader_grid_is_refused_by_name():
    """The front end's decimation convention and the model's anchor convention have to agree, and a
    disagreement produces a correctly-shaped tensor on the wrong grid."""
    with pytest.raises(ValueError, match="disagrees with the front end's total stride"):
        build_frontend(TINY_KWARGS, raw_per_step=8)


def test_a_raw_signal_of_the_wrong_length_is_refused_by_name():
    net = build_frontend(TINY_KWARGS)

    with pytest.raises(ValueError, match="expected a raw signal of"):
        net(torch.randn(2, 8 * SEQ_LEN), torch.ones(2, SEQ_LEN))


# ---------------------------------------------------------------------------------------
# The normaliser ban
# ---------------------------------------------------------------------------------------
def test_the_shipped_front_end_passes_the_normaliser_walk():
    """Every normaliser in the stack is an ``RMSNorm``, which reduces over the last axis only."""
    net = build_frontend(SHIPPED_KWARGS)

    refuse_time_pooling_norms(net)  # must not raise

    normalisers = [
        type(child).__name__
        for child in net.modules()
        if "Norm" in type(child).__name__ or "norm" in type(child).__name__
    ]
    assert normalisers and set(normalisers) == {"RMSNorm"}


@pytest.mark.parametrize("norm_type", TIME_POOLING_NORMS, ids=lambda t: t.__name__)
def test_every_banned_normaliser_family_is_actually_caught(norm_type):
    """Parametrised over the ban list itself, so a family added to the tuple without a working
    ``isinstance`` check cannot slip through -- ``SyncBatchNorm`` in particular does not subclass
    ``BatchNorm1d`` and would be missed by a check written for the concrete classes alone."""
    # GroupNorm alone takes the group count first; every other family takes the width.
    planted = nn.GroupNorm(1, 4) if norm_type is nn.GroupNorm else norm_type(4)
    holder = nn.Sequential(nn.Identity(), planted)

    with pytest.raises(ValueError, match=norm_type.__name__):
        refuse_time_pooling_norms(holder, label="planted")


def test_a_planted_group_norm_is_refused_at_construction(monkeypatch):
    """The negative control for the whole ban, and it is deliberately planted where a real edit
    would put one: inside the convolution block, reached only through the stage. If the walk ran at
    test time rather than at construction, this would build a leaking front end and report nothing.
    """

    class _LeakyBlock(GatedCausalConvBlock):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.leak = nn.GroupNorm(1, self.d_model)

    monkeypatch.setattr(frontend_module, "GatedCausalConvBlock", _LeakyBlock)

    with pytest.raises(ValueError, match="GroupNorm"):
        build_frontend(TINY_KWARGS)


def test_the_ban_names_the_offending_submodule_path():
    """A failure has to say *where*, or the message sends a reader looking through four stages."""
    holder = nn.Module()
    holder.stages = nn.ModuleList([nn.Sequential(nn.GroupNorm(1, 4))])

    with pytest.raises(ValueError, match=r"stages\.0\.0"):
        refuse_time_pooling_norms(holder)


def test_the_walk_accepts_a_stack_that_only_normalises_channels():
    """The paired half: a ban that rejected everything would also pass the tests above."""
    refuse_time_pooling_norms(nn.Sequential(RMSNorm(8), nn.LayerNorm(8), nn.Linear(8, 8)))


def test_the_front_end_module_is_runnable_as_a_table_printer():
    """The demoable outcome of this component, and a printer that raised would only be found by
    somebody running it."""
    frontend_module.main()


def test_a_shipped_frontend_carries_no_lag_attention_or_encoder_state():
    """The front end replaces the stored-feature adapters and nothing else; anything more here would
    move a difference downstream of the two encoder inputs, which is the one thing this package's
    comparison may not do."""
    net = build_frontend(SHIPPED_KWARGS)

    names = {type(child).__name__ for child in net.modules()}

    assert names <= {
        "CausalRawFrontend",
        "ModuleList",
        "CausalFrontendStage",
        "Linear",
        "GatedCausalConvBlock",
        "RMSNorm",
        "CausalDepthwiseConv1d",
        "Conv1d",
        "Dropout",
        "LayerScale",
        "CausalAntiAliasDecimate",
    }, f"unexpected submodules: {sorted(names)}"


def test_a_shipped_frontend_and_a_tiny_one_do_not_share_parameters():
    """Two independently parameterised front ends at identical settings is the design: sharing would
    make the source state a function of the target and destroy the purity the KL readout rests on."""
    first, second = build_frontend(TINY_KWARGS), build_frontend(TINY_KWARGS)

    shared = {id(parameter) for parameter in first.parameters()} & {
        id(parameter) for parameter in second.parameters()
    }

    assert not shared
