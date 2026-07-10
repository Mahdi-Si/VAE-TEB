r"""G0: the encoders' history states must not depend on the future.

:math:`K_t = \mathrm{KL}(q(z_t \mid Y_{\le t}, U_{\le t}) \,\|\, p(z_t \mid Y_{\le t}))` is a
transfer-entropy surrogate **only** if both distributions condition on the past. v1 applies
:class:`torch.nn.GroupNorm` to ``(B, C, T)`` tensors -- in ``CausalMultiChannelConvBlock``'s
``pre_norm`` and in ``_CausalConvLstmEncoder``'s ``stack_skip_norms`` -- and ``GroupNorm``
reduces over every non-batch dimension in a group, i.e. over :math:`(C/G, T)`. Its statistics
therefore pool across time and :math:`H_y[t]` becomes a function of :math:`Y_{>t}`.

``causal_norm=True`` swaps those modules for :class:`CausalGroupNorm`. These tests pin the
resulting invariant, and pin that the parity path (``causal_norm=False``) still reproduces
v1's leaky behaviour so Sprint-0 golden parity is untouched.

Note on the perturbation: it must be *random*, not a constant offset. Both encoders start with
a per-timestep :class:`torch.nn.LayerNorm`, which removes a uniform channel shift, so a
constant-offset probe would report a false pass.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
    CausalGroupNorm,
    SeqVaeLagAttnV3,
)

_LEAK_TOL = 1e-5  # float32 round-off on O(1) activations


def _future_leak(model: SeqVaeLagAttnV3, key: str, inputs, t0: int) -> float:
    r"""Max change in ``out[key][:, t0]`` when the strict future of the target is resampled."""
    y_st, y_ph, u = inputs
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        base = model(y_st, y_ph, u)[key]
        y_st2 = y_st.clone()
        y_st2[:, t0 + 1:, :] = torch.randn(
            y_st.shape[0], y_st.shape[1] - t0 - 1, y_st.shape[2], generator=g
        )
        perturbed = model(y_st2, y_ph, u)[key]
    return (base[:, t0] - perturbed[:, t0]).abs().max().item()


@pytest.mark.parametrize("key", ["target_state", "mu_prior", "logvar_prior", "decoder_state"])
def test_causal_norm_makes_the_target_history_state_causal(tiny_kwargs, inputs, key):
    model = SeqVaeLagAttnV3(causal_norm=True, **tiny_kwargs).eval()
    assert _future_leak(model, key, inputs, t0=8) < _LEAK_TOL, (
        f"{key}[t] still depends on y[>t] under causal_norm=True"
    )


def test_default_reproduces_v1_time_pooling(tiny_kwargs, inputs):
    """The parity path must keep v1's (leaky) GroupNorm, or golden parity would break."""
    model = SeqVaeLagAttnV3(causal_norm=False, **tiny_kwargs).eval()
    assert _future_leak(model, "target_state", inputs, t0=8) > 1e-3, (
        "the parity path is unexpectedly causal; test_v3_parity_v1 may be comparing "
        "against a modified v1"
    )


def test_causalize_replaces_exactly_the_encoder_norms(tiny_kwargs):
    causal = SeqVaeLagAttnV3(causal_norm=True, **tiny_kwargs)
    leaky = SeqVaeLagAttnV3(causal_norm=False, **tiny_kwargs)

    def count(module, cls):
        return sum(isinstance(m, cls) for m in module.modules())

    for enc in ("target_encoder", "source_encoder"):
        assert count(getattr(causal, enc), nn.GroupNorm) == 0
        assert count(getattr(causal, enc), CausalGroupNorm) == count(
            getattr(leaky, enc), nn.GroupNorm
        )
    assert causal.n_causalized_norms == 10
    # The horizon core pools over the forecast axis of a single anchor, not across input
    # time, so it is deliberately left alone.
    assert count(causal.horizon_core, nn.GroupNorm) == count(leaky.horizon_core, nn.GroupNorm)


def test_causal_group_norm_preserves_state_dict_keys_and_shapes(tiny_kwargs):
    """Warm-start from a v1 checkpoint must still align key-for-key (S4-T05)."""
    causal = SeqVaeLagAttnV3(causal_norm=True, **tiny_kwargs)
    leaky = SeqVaeLagAttnV3(causal_norm=False, **tiny_kwargs)
    sd_c, sd_l = causal.state_dict(), leaky.state_dict()

    assert set(sd_c) == set(sd_l)
    for k in sd_c:
        assert sd_c[k].shape == sd_l[k].shape, f"shape drift on {k}"
    # And the leaky weights load straight into the causal model.
    causal.load_state_dict(sd_l, strict=True)


def test_causal_group_norm_matches_groupnorm_applied_per_timestep():
    r"""``CausalGroupNorm`` is exactly ``GroupNorm`` with the time axis folded into the batch."""
    torch.manual_seed(0)
    B, C, T, G = 3, 16, 7, 4
    x = torch.randn(B, C, T)

    causal = CausalGroupNorm(G, C)
    with torch.no_grad():
        causal.weight.normal_()
        causal.bias.normal_()

    reference = nn.GroupNorm(G, C)
    with torch.no_grad():
        reference.weight.copy_(causal.weight)
        reference.bias.copy_(causal.bias)
        got = causal(x)
        # Fold time into batch so GroupNorm cannot pool across it.
        folded = x.permute(0, 2, 1).reshape(B * T, C, 1)
        want = reference(folded).reshape(B, T, C).permute(0, 2, 1)

    assert torch.allclose(got, want, atol=1e-5)


def test_causal_group_norm_output_at_t_ignores_other_timesteps():
    torch.manual_seed(0)
    norm = CausalGroupNorm(4, 16)
    x = torch.randn(2, 16, 9)
    x2 = x.clone()
    x2[:, :, 5:] = torch.randn(2, 16, 4)
    with torch.no_grad():
        assert torch.allclose(norm(x)[:, :, 4], norm(x2)[:, :, 4], atol=1e-6)


def test_causal_group_norm_rejects_indivisible_channels():
    with pytest.raises(ValueError, match="divisible"):
        CausalGroupNorm(num_groups=3, num_channels=16)
