"""S1-T04: zero KL at initialization under the production path (residual + smooth).

The headline invariant: with ``posterior_logvar='residual'`` and ``logvar_bound='smooth'``,
the per-step KL is exactly zero at init (mu_post == mu_prior and logvar_post == logvar_prior),
for both the flat and head-structured latents.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


@pytest.mark.parametrize("head_structured", [False, True])
def test_zero_kl_at_init(tiny_kwargs, inputs, head_structured):
    model = SeqVaeLagAttnV3(
        **dict(
            tiny_kwargs,
            head_structured_latent=head_structured,
            posterior_logvar="residual",
            logvar_bound="smooth",
            kld_support="anchor",
        )
    ).eval()
    out = model(*inputs)
    kmax = out["kld_per_t"].abs().max().item()
    assert kmax < 1e-6, (
        f"KL not zero at init under residual+smooth (head_structured={head_structured}): "
        f"max|kld_per_t| = {kmax:.3e}"
    )
    if head_structured:
        # Per-head KL is also identically zero.
        assert out["kld_per_t_per_head"].abs().max().item() < 1e-6
