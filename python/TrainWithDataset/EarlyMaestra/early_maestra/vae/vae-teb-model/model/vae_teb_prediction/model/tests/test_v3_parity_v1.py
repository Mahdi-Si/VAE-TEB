"""S0-T02: golden parity -- v3 == v1 under the parity configuration.

With ``posterior_logvar='independent'``, ``logvar_bound='clamp'``, ``kld_support='full'``,
``sigma_obs=1.0``, ``lag_bias_init='normal'`` and seed-matched reparameterization noise,
every shared forward tensor and every ``compute_loss`` term must match v1 to ``< 1e-5``.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3

_TOL = 1e-5
# Loss terms shared with v1 (v3 additionally reports kld_raw / kld_train / kld_active_frac).
_SHARED_LOSS_KEYS = (
    "feat_loss", "base_loss", "kld_loss", "total_loss",
    "mean_logvar_full", "mean_logvar_base", "lag_smoothness",
)


def _parity_kwargs(tiny_kwargs):
    return dict(
        tiny_kwargs,
        posterior_logvar="independent",
        logvar_bound="clamp",
        kld_support="full",
        lag_bias_init="normal",
    )


def _build_pair(tiny_kwargs):
    torch.manual_seed(1)
    v1 = SeqVaeLagAttnV1(**tiny_kwargs).eval()
    torch.manual_seed(1)
    v3 = SeqVaeLagAttnV3(**_parity_kwargs(tiny_kwargs)).eval()
    return v1, v3


def _max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def test_forward_tensor_parity(tiny_kwargs, inputs):
    v1, v3 = _build_pair(tiny_kwargs)

    torch.manual_seed(2)
    o1 = v1(*inputs)
    torch.manual_seed(2)
    o3 = v3(*inputs)

    for k, v in o1.items():
        if not isinstance(v, torch.Tensor):
            continue
        diff = _max_abs_diff(o3[k], v)
        assert diff < _TOL, f"forward tensor {k!r} differs by {diff:.3e} (>= {_TOL})"


def test_compute_loss_parity(tiny_kwargs, inputs):
    v1, v3 = _build_pair(tiny_kwargs)

    torch.manual_seed(2)
    o1 = v1(*inputs)
    torch.manual_seed(2)
    o3 = v3(*inputs)

    loss_kwargs = dict(
        beta=0.1,
        likelihood="gaussian_nll",
        sigma_obs=1.0,
        free_bits=0.1,
        detach_baseline_in_full=True,
        lambda_lag=1e-3,
    )
    l1 = v1.compute_loss(o1, inputs[0], inputs[1], **loss_kwargs)
    l3 = v3.compute_loss(o3, inputs[0], inputs[1], **loss_kwargs)

    for k in _SHARED_LOSS_KEYS:
        diff = _max_abs_diff(l3[k], l1[k])
        assert diff < _TOL, f"loss term {k!r} differs by {diff:.3e} (>= {_TOL})"


def test_parity_holds_for_head_structured(tiny_kwargs, inputs):
    kw = dict(tiny_kwargs, head_structured_latent=True)
    v1, v3 = _build_pair(kw)

    torch.manual_seed(2)
    o1 = v1(*inputs)
    torch.manual_seed(2)
    o3 = v3(*inputs)

    for k in ("mu_post", "logvar_post", "kld_per_t", "kld_per_t_per_head", "te_lag_map"):
        diff = _max_abs_diff(o3[k], o1[k])
        assert diff < _TOL, f"{k!r} differs by {diff:.3e} (head-structured)"
