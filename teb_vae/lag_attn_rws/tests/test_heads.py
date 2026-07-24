r"""The full-latent prior head: three outputs, no dead parameters, exact bound identity.

The head exists instead of reusing the sibling's ``PriorHead`` because that one also emits a
``decoder_state`` this architecture must not have: reusing it and discarding the tensor would
leave dead parameters that a distributed run must then be told to tolerate. So the tests here
pin the *absence* as much as the outputs.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import BATCH, SEQ_LEN

_D_MODEL, _D_Z = 32, 8
_CLAMP = (-5.0, 3.0)


def _head() -> FullLatentPriorHead:
    torch.manual_seed(0)
    return FullLatentPriorHead(
        d_model=_D_MODEL, d_z=_D_Z, logvar_clamp=_CLAMP, dropout=0.0, mu_scale=5.0
    )


def _state() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, _D_MODEL, generator=torch.Generator().manual_seed(1))


def test_the_head_returns_three_latent_shaped_tensors():
    mu, logvar, raw_logvar = _head()(_state())
    for tensor in (mu, logvar, raw_logvar):
        assert tensor.shape == (BATCH, SEQ_LEN, _D_Z)


def test_the_bounded_logvar_is_exactly_the_bound_of_the_raw_one():
    """The posterior residual is applied to the raw value; if this identity drifted, the
    zero-delta posterior would no longer reproduce the prior exactly."""
    _, logvar, raw_logvar = _head()(_state())
    assert torch.equal(logvar, smooth_bound(raw_logvar, *_CLAMP))
    assert (logvar > _CLAMP[0]).all() and (logvar < _CLAMP[1]).all()


def test_the_prior_mean_respects_its_saturation_bound():
    mu, _, _ = _head()(_state())
    assert (mu.abs() <= 5.0).all()


def test_no_parameter_is_dead():
    """Every parameter must reach an output; a dead head is the exact failure reusing the
    sibling's PriorHead would have produced."""
    head = _head()
    mu, logvar, _ = head(_state())
    grads = torch.autograd.grad(
        mu.sum() + logvar.sum(), list(head.parameters()), allow_unused=True
    )
    dead = [
        name
        for (name, _), grad in zip(head.named_parameters(), grads)
        if grad is None
    ]
    assert not dead, f"parameters with no path to an output: {dead}"


def test_there_is_no_decoder_state_pathway():
    head = _head()
    assert not hasattr(head, "decoder_state_head")
    assert not hasattr(head, "dec_input_norm")


def test_a_non_positive_mu_scale_is_rejected():
    with pytest.raises(ValueError, match="mu_scale"):
        FullLatentPriorHead(d_model=_D_MODEL, d_z=_D_Z, mu_scale=0.0)


def test_perturb_posterior_moves_the_posterior_off_the_prior(
    tiny_kwargs, inputs, perturb_posterior
):
    """The shared perturbation fixture must bite on this model, or every KL assertion in the
    suite is vacuous (at init the posterior equals the prior exactly)."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**tiny_kwargs).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    assert (out["mu_post"] - out["mu_prior"]).abs().max().item() > 1e-6
