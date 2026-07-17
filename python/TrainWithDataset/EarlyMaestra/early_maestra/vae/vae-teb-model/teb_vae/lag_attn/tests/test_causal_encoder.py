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


def _make_encoder(kernels=_TARGET_KERNELS, causal: bool = True) -> CausalConvLstmEncoder:
    torch.manual_seed(0)
    encoder = CausalConvLstmEncoder(
        d_model=_D_MODEL,
        cnn_kernels=kernels,
        cnn_dilations=_DILATIONS,
        lstm_layers=2,
        lstm_dropout=0.0,
        conv_dropout=0.0,
    )
    if causal:
        causalize_norms(encoder)
    return encoder.eval()


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
    for in_dim in (87, 101, 58):
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
