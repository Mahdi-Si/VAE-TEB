"""S1-T02 / S1-T03: posterior log-variance residual head + zero-init.

In ``posterior_logvar='residual'`` mode the posterior builds a zero-initialised
``delta_logvar_head`` (mirroring ``delta_mu_head``) and drops the independent v1
``logvar_post_head``. With the delta head zeroed, the posterior log-variance equals the
prior's bounded log-variance exactly, for both the flat and head-structured latents.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


def _residual_model(tiny_kwargs, head_structured):
    return SeqVaeLagAttnV3(
        **dict(
            tiny_kwargs,
            head_structured_latent=head_structured,
            posterior_logvar="residual",
            logvar_bound="smooth",
        )
    ).eval()


def _delta_logvar_is_zero(head) -> bool:
    dlh = head.delta_logvar_head
    layers = list(dlh) if isinstance(dlh, nn.ModuleList) else [dlh]
    for layer in layers:
        if layer.weight.abs().max().item() != 0.0:
            return False
        if layer.bias is not None and layer.bias.abs().max().item() != 0.0:
            return False
    return True


def test_residual_mode_builds_delta_logvar_and_drops_independent_head(tiny_kwargs):
    model = _residual_model(tiny_kwargs, head_structured=False)
    ph = model.posterior_head
    assert hasattr(ph, "delta_logvar_head")
    assert not hasattr(ph, "logvar_post_head"), "independent logvar head should be removed"


def test_delta_logvar_head_zero_initialised(tiny_kwargs):
    for hs in (False, True):
        model = _residual_model(tiny_kwargs, head_structured=hs)
        assert _delta_logvar_is_zero(model.posterior_head), (
            f"delta_logvar_head not zero-initialised (head_structured={hs})"
        )


def test_logvar_post_equals_prior_at_init(tiny_kwargs, inputs):
    # With delta heads zeroed, logvar_post == logvar_prior and mu_post == mu_prior exactly.
    for hs in (False, True):
        model = _residual_model(tiny_kwargs, head_structured=hs)
        out = model(*inputs)
        lv_gap = (out["logvar_post"] - out["logvar_prior"]).abs().max().item()
        mu_gap = (out["mu_post"] - out["mu_prior"]).abs().max().item()
        assert lv_gap < 1e-6, f"logvar_post != logvar_prior at init (hs={hs}): {lv_gap:.3e}"
        assert mu_gap < 1e-6, f"mu_post != mu_prior at init (hs={hs}): {mu_gap:.3e}"
