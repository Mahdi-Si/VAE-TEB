r"""At initialisation the model must assert that the source says nothing.

The posterior is a residual on the prior and both delta heads are zero-initialised, so at init
$q \equiv p$ and $K_t \equiv 0$ *exactly* -- not approximately. Training then has to earn every
nat of coupling it later reports, against a null the model started from.

This also documents the trap that shapes the rest of this suite: because $K_t$ is identically
zero here, **any** KL assertion on a freshly-built model passes, including on a model that is
completely wrong. Every other KL test perturbs the posterior first. This is the one file where
the zero is the point.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

_TOL = 1e-6


def _model(prod_kwargs, **overrides):
    torch.manual_seed(0)
    return SeqVaeLagAttn(**dict(prod_kwargs, **overrides)).eval()


@pytest.mark.parametrize("head_structured", [False, True])
def test_kld_per_t_is_zero_at_init(prod_kwargs, inputs, head_structured):
    model = _model(prod_kwargs, head_structured_latent=head_structured)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert out["kld_per_t"].abs().max().item() < _TOL


@pytest.mark.parametrize("head_structured", [False, True])
def test_the_posterior_equals_the_prior_at_init(prod_kwargs, inputs, head_structured):
    model = _model(prod_kwargs, head_structured_latent=head_structured)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)

    mu_gap = (out["mu_post"] - out["mu_prior"]).abs().max().item()
    logvar_gap = (out["logvar_post"] - out["logvar_prior"]).abs().max().item()
    assert mu_gap < _TOL, f"mu_post != mu_prior at init (head_structured={head_structured})"
    assert logvar_gap < _TOL, f"logvar_post != logvar_prior (head_structured={head_structured})"


@pytest.mark.parametrize("head_structured", [False, True])
def test_the_delta_heads_are_zeroed(prod_kwargs, head_structured):
    model = _model(prod_kwargs, head_structured_latent=head_structured)
    for name in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(model.posterior_head, name)
        layers = list(module) if isinstance(module, nn.ModuleList) else [module]
        for layer in layers:
            assert layer.weight.abs().max().item() == 0.0, f"{name} weight not zeroed"
            if layer.bias is not None:
                assert layer.bias.abs().max().item() == 0.0, f"{name} bias not zeroed"


def test_the_residual_decoder_mean_head_is_zeroed(prod_kwargs):
    """So the full forecast starts equal to the baseline, and divergence is learned."""
    head = _model(prod_kwargs).residual_decoder.mean_head
    assert head.weight.abs().max().item() == 0.0
    assert head.bias is not None and head.bias.abs().max().item() == 0.0


def test_the_full_forecast_equals_the_baseline_at_init(prod_kwargs, inputs):
    model = _model(prod_kwargs)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert torch.equal(out["mu_full"], out["mu_base"])
    assert out["delta_mu_src"].abs().max().item() == 0.0


def test_the_kl_becomes_nonzero_once_perturbed(prod_kwargs, inputs, perturb_posterior):
    """The zero must be a property of the init, not of the model being unable to produce a KL.

    Without this, a model whose KL was structurally stuck at zero -- a broken posterior, a
    detached graph -- would pass every test above.
    """
    model = _model(prod_kwargs)
    perturb_posterior(model)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert out["kld_per_t"].abs().max().item() > _TOL


def test_the_zero_survives_the_generic_weight_init(prod_kwargs):
    """The delta heads are zeroed *after* the generic init, which would otherwise refill them."""
    model = _model(prod_kwargs, init_weights=True)
    assert model.posterior_head.delta_mu_head.weight.abs().max().item() == 0.0
