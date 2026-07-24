r"""The encoders' history states must not depend on the future.

$K_t = \mathrm{KL}(q(z_t \mid Y_{\le t}, U_{\le t}) \,\|\, p(z_t \mid Y_{\le t}))$ is a
transfer-entropy surrogate **only** if both distributions condition on the past. The convolution
blocks and the inter-block skips normalise ``(B, C, T)`` tensors with ``torch.nn.GroupNorm``,
which reduces over every non-batch dimension in a group -- i.e. over $(C/G, T)$. Its statistics
therefore pool across time, and $H^y[t]$ silently becomes a function of $Y_{>t}$.

Nothing about that leak is visible in a loss curve. It makes the model *better* at forecasting,
which is exactly why it has to be tested rather than watched for: the number it corrupts is the
one the model exists to produce.

:func:`causalize_norms` swaps those modules for :class:`CausalGroupNorm`. These tests pin the
resulting invariant, and pin that the swap is structurally free -- same keys, same shapes.

Note on the perturbation: it must be *random*, not a constant offset. The encoder starts with a
per-timestep ``LayerNorm``, which removes a uniform channel shift, so a constant-offset probe
would report a false pass on a leaky model.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import CausalGroupNorm, causalize_norms
from teb_vae.lag_attn.nets.encoders import CausalConvLstmEncoder, InputAdapter

_LEAK_TOL = 1e-5  # float32 round-off on O(1) activations

# The kernel schedules the two streams actually run with.
_TARGET_KERNELS = (3, 7, 11)
_SOURCE_KERNELS = (3, 5, 11)
_DILATIONS = (1, 2, 4)

_D_MODEL = 32
_BATCH, _SEQ_LEN = 2, 16


def _make_encoder(
    kernels=_TARGET_KERNELS,
    causal: bool = True,
    stack_skip_connection: bool = True,
    post_residual_activation: bool = True,
    conv_norm_groups=None,
) -> CausalConvLstmEncoder:
    torch.manual_seed(0)
    encoder = CausalConvLstmEncoder(
        d_model=_D_MODEL,
        cnn_kernels=kernels,
        cnn_dilations=_DILATIONS,
        lstm_layers=2,
        lstm_dropout=0.0,
        conv_dropout=0.0,
        stack_skip_connection=stack_skip_connection,
        post_residual_activation=post_residual_activation,
        conv_norm_groups=conv_norm_groups,
    )
    if causal:
        causalize_norms(encoder)
    return encoder.eval()


def _seam_gate_attenuation(mlp, in_dim: int, seed: int = 0) -> float:
    """Backward attenuation of a residual-MLP seam's final activation gate.

    Returns the RMS of the cotangent just *below* the final activation over the cotangent just
    *above* it: how much the post-residual GELU shrinks the gradient passing back through the seam.
    ``1.0`` when the seam carries no final activation (gradient-transparent), which is what the flag
    off produces.
    """
    if mlp.final_act is None:
        return 1.0
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 6, in_dim, generator=generator)
    captured: dict = {}
    handle = mlp.final_act.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("pre", inputs[0].detach())
    )
    with torch.no_grad():
        mlp(x)
    handle.remove()
    pre = captured["pre"].clone().requires_grad_(True)
    out = mlp.final_act(pre)
    grad_out = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed + 1))
    (grad_pre,) = torch.autograd.grad(out, pre, grad_outputs=grad_out)
    return (grad_pre.pow(2).mean().sqrt() / grad_out.pow(2).mean().sqrt()).item()


def _conv_stack_growth(encoder: CausalConvLstmEncoder) -> float:
    """Stack-output RMS over stack-input RMS through the dilated conv stack.

    Mirrors the loop in :meth:`CausalConvLstmEncoder.forward` because the growth it measures is
    hidden by the ``output_norm`` LayerNorm at the encoder exit -- it can only be seen on the
    intermediate stream.
    """
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=generator)
    with torch.no_grad():
        x_lin = encoder.front_mlp(x)
        x_conv = x_lin.transpose(1, 2).contiguous()
        stack_in_rms = x_conv.pow(2).mean().sqrt()
        out = encoder.convs[0](x_conv)
        for index in range(1, len(encoder.convs)):
            block_out = encoder.convs[index](out)
            if encoder.stack_skip_norms is not None:
                block_out = block_out + encoder.stack_skip_norms[index - 1](out)
            out = block_out
        stack_out_rms = out.pow(2).mean().sqrt()
    return (stack_out_rms / stack_in_rms).item()


def _future_leak(encoder: CausalConvLstmEncoder, t0: int) -> float:
    """Max change in ``out[:, t0]`` when the strict future of the input is resampled."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=generator)

    perturbed_input = x.clone()
    perturbed_input[:, t0 + 1 :, :] = torch.randn(
        _BATCH, _SEQ_LEN - t0 - 1, _D_MODEL, generator=torch.Generator().manual_seed(11)
    )

    with torch.no_grad():
        base = encoder(x)
        perturbed = encoder(perturbed_input)
    return (base[:, t0] - perturbed[:, t0]).abs().max().item()


def test_causalized_encoder_ignores_the_future():
    assert _future_leak(_make_encoder(causal=True), t0=8) < _LEAK_TOL


def test_the_leak_probe_can_detect_a_leak():
    """The probe above is only worth having if a leaky encoder fails it.

    Without this, a probe that always returned $0$ -- a broken perturbation, a wrong index --
    would make the causality test above pass on any model at all.
    """
    assert _future_leak(_make_encoder(causal=False), t0=8) > 1e-3


def test_causalize_replaces_every_group_norm_in_both_encoders():
    target = _make_encoder(_TARGET_KERNELS, causal=False)
    source = _make_encoder(_SOURCE_KERNELS, causal=False)

    def count(module: nn.Module, cls: type) -> int:
        return sum(isinstance(child, cls) for child in module.modules())

    leaky_total = count(target, nn.GroupNorm) + count(source, nn.GroupNorm)
    replaced = causalize_norms(target) + causalize_norms(source)

    # Five per encoder: one pre_norm inside each of the three conv blocks, plus the two
    # inter-block skip norms.
    assert leaky_total == 10
    assert replaced == 10
    assert count(target, nn.GroupNorm) == 0
    assert count(source, nn.GroupNorm) == 0
    assert count(target, CausalGroupNorm) + count(source, CausalGroupNorm) == 10


def test_causalize_is_idempotent():
    """A second pass finds nothing left to replace."""
    encoder = _make_encoder(causal=False)
    assert causalize_norms(encoder) == 5
    assert causalize_norms(encoder) == 0


def test_causal_and_leaky_encoders_share_state_dict_keys_and_shapes():
    """The swap is structurally free: it changes what is pooled, not what is stored."""
    causal = _make_encoder(causal=True)
    leaky = _make_encoder(causal=False)

    causal_sd, leaky_sd = causal.state_dict(), leaky.state_dict()
    assert set(causal_sd) == set(leaky_sd)
    for key in causal_sd:
        assert causal_sd[key].shape == leaky_sd[key].shape, f"shape drift on {key}"

    # And the leaky weights load straight into the causal model.
    causal.load_state_dict(leaky_sd, strict=True)


def test_causal_group_norm_matches_groupnorm_applied_per_timestep():
    r"""``CausalGroupNorm`` is exactly ``GroupNorm`` with the time axis folded into the batch."""
    torch.manual_seed(0)
    batch, channels, seq_len, groups = 3, 16, 7, 4
    x = torch.randn(batch, channels, seq_len)

    causal = CausalGroupNorm(groups, channels)
    with torch.no_grad():
        causal.weight.normal_()
        causal.bias.normal_()

    reference = nn.GroupNorm(groups, channels)
    with torch.no_grad():
        reference.weight.copy_(causal.weight)
        reference.bias.copy_(causal.bias)
        got = causal(x)
        # Fold time into batch so GroupNorm cannot pool across it.
        folded = x.permute(0, 2, 1).reshape(batch * seq_len, channels, 1)
        want = reference(folded).reshape(batch, seq_len, channels).permute(0, 2, 1)

    assert torch.allclose(got, want, atol=1e-5)


def test_causal_group_norm_output_at_t_ignores_other_timesteps():
    torch.manual_seed(0)
    norm = CausalGroupNorm(4, 16)
    x = torch.randn(2, 16, 9)
    perturbed = x.clone()
    perturbed[:, :, 5:] = torch.randn(2, 16, 4)
    with torch.no_grad():
        assert torch.allclose(norm(x)[:, :, 4], norm(perturbed)[:, :, 4], atol=1e-6)


def test_causal_group_norm_rejects_indivisible_channels():
    with pytest.raises(ValueError, match="divisible"):
        CausalGroupNorm(num_groups=3, num_channels=16)


def test_input_adapter_projects_both_stream_widths():
    """One adapter, two widths. The stream is an argument, not a class."""
    for in_dim in (109, 58, 15):
        adapter = InputAdapter(in_dim=in_dim, d_model=_D_MODEL, dropout=0.0).eval()
        x = torch.randn(_BATCH, _SEQ_LEN, in_dim)
        with torch.no_grad():
            assert adapter(x).shape == (_BATCH, _SEQ_LEN, _D_MODEL)


def test_encoder_rejects_mismatched_schedules():
    with pytest.raises(ValueError, match="equal length"):
        CausalConvLstmEncoder(
            d_model=_D_MODEL,
            cnn_kernels=(3, 5),
            cnn_dilations=(1,),
            lstm_layers=1,
            lstm_dropout=0.0,
            conv_dropout=0.0,
        )


def test_encoder_rejects_an_empty_conv_stack():
    with pytest.raises(ValueError, match="at least one"):
        CausalConvLstmEncoder(
            d_model=_D_MODEL,
            cnn_kernels=(),
            cnn_dilations=(),
            lstm_layers=1,
            lstm_dropout=0.0,
            conv_dropout=0.0,
        )


# ---------------------------------------------------------------------------------------
# The plain residual conv stack (stack_skip_connection=False)
# ---------------------------------------------------------------------------------------
def test_the_plain_stack_builds_no_inter_block_skip_norms():
    """Off, the redundant term is gone -- and so are its GroupNorms, which would otherwise sit in
    DDP's expectation set as starved parameters."""
    encoder = _make_encoder(causal=False, stack_skip_connection=False)
    assert encoder.stack_skip_norms is None
    # Only the per-block conv pre-norms survive; the two inter-block skip norms are not built.
    group_norms = sum(isinstance(module, nn.GroupNorm) for module in encoder.modules())
    assert group_norms == len(_DILATIONS)


def test_the_plain_stack_leaves_no_orphan_parameter():
    """The DDP consequence of not building the skip norms: one backward gives every trainable
    parameter a gradient, so plain ``'ddp'`` (find_unused_parameters=False) stays valid."""
    encoder = _make_encoder(causal=False, stack_skip_connection=False).train()
    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL)
    encoder(x).pow(2).sum().backward()
    starved = [
        name for name, param in encoder.named_parameters() if param.requires_grad and param.grad is None
    ]
    assert not starved, f"parameters expecting a gradient but not receiving one: {starved}"


def test_the_plain_stack_is_still_strictly_causal():
    """Dropping the second residual changes the activation scale, not the causal structure: the
    per-block causal convolutions and the causalised norms still make the state at t a function of
    the past only."""
    assert _future_leak(_make_encoder(causal=True, stack_skip_connection=False), t0=8) < _LEAK_TOL


def test_the_plain_stack_holds_activation_scale_and_the_double_stack_does_not():
    """The D1 defect and its fix, as a standing guard. The double residual injects a
    GroupNorm-rescaled second copy of the stream at every stage onto an un-renormalised, growing
    stream, so activation RMS inflates through depth; the single clean residual chain holds it near
    1x. Self-verifying: the flag-on control must exceed the bound the flag-off path stays under, so
    a refactor cannot make both pass by breaking the probe."""
    plain = _conv_stack_growth(_make_encoder(causal=False, stack_skip_connection=False))
    double = _conv_stack_growth(_make_encoder(causal=False, stack_skip_connection=True))
    assert plain < 2.0, f"the plain stack inflated activations {plain:.2f}x (bound 2.0x)"
    assert double > 2.5, (
        f"the double stack only grew {double:.2f}x; the negative control no longer bites, so the "
        f"bound above is not proving anything"
    )


# ---------------------------------------------------------------------------------------
# Plain residual seams (post_residual_activation=False)
# ---------------------------------------------------------------------------------------
def test_plain_seams_change_the_encoder_output():
    """Dropping the post-residual GELU at the front and fusion seams changes what the encoder
    computes -- it is not a silent no-op."""
    on = _make_encoder(causal=False, post_residual_activation=True)
    off = _make_encoder(causal=False, post_residual_activation=False)
    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert not torch.allclose(on(x), off(x))


def test_plain_seams_leave_no_orphan_parameter():
    """The final LayerNorm and GELU are removed, not merely bypassed, so no parameter is left
    unused: one backward gives every trainable parameter a gradient."""
    encoder = _make_encoder(causal=False, post_residual_activation=False).train()
    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL)
    encoder(x).pow(2).sum().backward()
    starved = [
        name for name, param in encoder.named_parameters() if param.requires_grad and param.grad is None
    ]
    assert not starved, f"parameters expecting a gradient but not receiving one: {starved}"


def test_plain_seams_preserve_causality():
    """Removing the seam activation changes the representation's statistics, not its causal
    structure."""
    assert _future_leak(_make_encoder(causal=True, post_residual_activation=False), t0=8) < _LEAK_TOL


def test_plain_seams_drop_one_layernorm_per_affected_residual_mlp():
    """``final_activation=False`` removes the final LayerNorm from each affected ResidualMLP -- one
    per adapter (its ``res_mlp``) and two per encoder (``front_mlp`` and ``fusion``). At the
    production width d_model=128 that is 256 parameters each."""
    def adapter(post: bool) -> int:
        torch.manual_seed(0)
        return sum(
            p.numel()
            for p in InputAdapter(in_dim=58, d_model=128, dropout=0.0,
                                  post_residual_activation=post).parameters()
        )

    def encoder(post: bool) -> int:
        torch.manual_seed(0)
        return sum(
            p.numel()
            for p in CausalConvLstmEncoder(
                d_model=128, cnn_kernels=_TARGET_KERNELS, cnn_dilations=_DILATIONS,
                lstm_layers=2, lstm_dropout=0.0, conv_dropout=0.0,
                post_residual_activation=post,
            ).parameters()
        )

    assert adapter(True) - adapter(False) == 256           # one LayerNorm(128)
    assert encoder(True) - encoder(False) == 512           # two LayerNorm(128)


def test_the_seam_activation_gates_the_backward_gradient_and_removing_it_does_not():
    """The R1 finding, as a standing guard. The post-residual GELU shrinks the gradient passing
    back through each seam; without it the seam is gradient-transparent. Flag on: every seam
    attenuates (ratio < 0.9). Flag off: every seam passes the gradient through unchanged
    (ratio 1.0, inside [0.9, 1.1])."""
    on_adapter = InputAdapter(in_dim=58, d_model=_D_MODEL, dropout=0.0,
                              post_residual_activation=True).eval()
    off_adapter = InputAdapter(in_dim=58, d_model=_D_MODEL, dropout=0.0,
                               post_residual_activation=False).eval()
    on_encoder = _make_encoder(causal=False, post_residual_activation=True)
    off_encoder = _make_encoder(causal=False, post_residual_activation=False)

    on_ratios = {
        # The adapter projects 58 -> d_model before its res_mlp, so the seam runs at d_model width.
        "adapter": _seam_gate_attenuation(on_adapter.res_mlp, _D_MODEL),
        "front": _seam_gate_attenuation(on_encoder.front_mlp, _D_MODEL),
        "fusion": _seam_gate_attenuation(on_encoder.fusion, 2 * _D_MODEL),
    }
    off_ratios = {
        "adapter": _seam_gate_attenuation(off_adapter.res_mlp, _D_MODEL),
        "front": _seam_gate_attenuation(off_encoder.front_mlp, _D_MODEL),
        "fusion": _seam_gate_attenuation(off_encoder.fusion, 2 * _D_MODEL),
    }
    for seam, ratio in on_ratios.items():
        assert ratio < 0.9, f"{seam} seam does not gate the gradient with the flag on: {ratio:.3f}"
    for seam, ratio in off_ratios.items():
        assert 0.9 <= ratio <= 1.1, f"{seam} seam is not transparent with the flag off: {ratio:.3f}"


# ---------------------------------------------------------------------------------------
# Conv pre-norm group count (conv_norm_groups)
# ---------------------------------------------------------------------------------------
def test_conv_norm_groups_sets_every_conv_pre_norm_group_count():
    """``conv_norm_groups=1`` builds ``GroupNorm(1, d_model)`` conv pre-norms -- a per-timestep
    normaliser over all channels -- in place of the default ``min(8, d_model)``. The Conv1d group
    count is a separate argument and is untouched."""
    default = _make_encoder(causal=False)
    grouped = _make_encoder(causal=False, conv_norm_groups=1)

    for block in default.convs:
        assert block.pre_norm.num_groups == min(8, _D_MODEL)
    for block in grouped.convs:
        assert block.pre_norm.num_groups == 1
        assert block.pre_norm.num_channels == _D_MODEL


def test_conv_norm_groups_changes_num_groups_not_the_norm_count_or_parameters():
    """The flag changes ``num_groups``, not the module count: ``causalize_norms`` still replaces the
    same number of norms, and the parameter count is unchanged (GroupNorm affine params are
    per-channel, independent of the group count), so nothing is starved either way."""
    default = _make_encoder(_TARGET_KERNELS, causal=False)
    grouped = _make_encoder(_TARGET_KERNELS, causal=False, conv_norm_groups=1)

    assert causalize_norms(default) == causalize_norms(grouped)
    assert sum(p.numel() for p in default.parameters()) == sum(
        p.numel() for p in grouped.parameters()
    )


def test_conv_norm_groups_default_is_the_untouched_min8_norm():
    """``None`` is the untouched default: an explicit ``conv_norm_groups=None`` encoder is fixed-seed
    identical to one built before the flag existed."""
    explicit = _make_encoder(causal=False, conv_norm_groups=None)
    reference = _make_encoder(causal=False)  # no conv_norm_groups argument at all
    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        assert torch.equal(explicit(x), reference(x))
