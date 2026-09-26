r"""At initialisation the model must assert that the source says nothing.

The posterior is a residual on the prior and both delta heads are zero-initialised, so at init
$q \equiv p$ and $K_t \equiv 0$ *exactly* -- not approximately. Training then has to earn every
nat of coupling it later reports, against a null the model started from.

This also documents the trap that shapes the rest of this suite: because $K_t$ is identically
zero here, **any** KL assertion on a freshly-built model passes, including on a model that is
completely wrong. Every other KL test perturbs the posterior first. This is the one file where
the zero is the point. (That the full forecast starts equal to the baseline is pinned beside the
FiLM init in ``test_logvar_floor.py``.)
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

_TOL = 1e-6


def _model(prod_kwargs, **overrides):
    torch.manual_seed(0)
    return SeqVaeLagAttn(**dict(prod_kwargs, **overrides)).eval()


@pytest.mark.parametrize("head_structured", [False, True])
def test_the_posterior_equals_the_prior_and_the_kl_is_zero_at_init(
    prod_kwargs, inputs, head_structured
):
    """Through the default constructor, which runs the generic weight init first: the delta heads
    must be zeroed *after* it, or it would refill them."""
    model = _model(prod_kwargs, head_structured_latent=head_structured)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)

    mu_gap = (out["mu_post"] - out["mu_prior"]).abs().max().item()
    logvar_gap = (out["logvar_post"] - out["logvar_prior"]).abs().max().item()
    assert mu_gap < _TOL, f"mu_post != mu_prior at init (head_structured={head_structured})"
    assert logvar_gap < _TOL, f"logvar_post != logvar_prior (head_structured={head_structured})"
    assert out["kld_per_t"].abs().max().item() < _TOL


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


