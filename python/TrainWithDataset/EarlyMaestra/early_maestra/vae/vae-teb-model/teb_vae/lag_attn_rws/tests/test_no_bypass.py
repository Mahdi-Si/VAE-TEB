r"""Gradient reaches the decoder only through $z$ -- the claim made operational.

"No decoder bypass" is what forces the prior latent to carry the FHR predictive state: if any
target-derived tensor reached the decoder around the latent, $z$ could shrink back into a
residual code and every downstream reading of $\mu^p$ as "the fetal state" would be
unsupported. Rhetoric is cheap here, so the test is autograd: with the latent samples detached
inside the model's own forward, no gradient path from the base forecast to the target encoder
may survive. Both directions are asserted -- the same probe without the detach must find
gradients -- so the test cannot pass vacuously on a broken graph.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _model(tiny_kwargs) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**tiny_kwargs)


def _grads_to_target_encoder(model: SeqVaeLagAttnRws, out: dict):
    return torch.autograd.grad(
        out["mu_base"].sum(),
        list(model.target_encoder.parameters()),
        allow_unused=True,
    )


def test_with_z_detached_no_gradient_reaches_the_target_encoder(tiny_kwargs, inputs):
    """The detach happens inside the model's real forward -- by wrapping the sampling method --
    so the probe covers the wiring as-built, not a hand-assembled call into the decoder."""
    model = _model(tiny_kwargs)
    sample = model._reparameterize_shared
    model._reparameterize_shared = lambda *args: tuple(  # type: ignore[method-assign]
        z.detach() for z in sample(*args)
    )

    out = model(*inputs)
    grads = _grads_to_target_encoder(model, out)
    leaked = [
        name
        for (name, _), grad in zip(model.target_encoder.named_parameters(), grads)
        if grad is not None
    ]
    assert not leaked, f"target-encoder parameters reachable around z: {leaked}"


def test_without_the_detach_the_same_probe_finds_gradients(tiny_kwargs, inputs):
    """The positive direction: through z, every target-encoder parameter is on the path."""
    model = _model(tiny_kwargs)
    out = model(*inputs)
    grads = _grads_to_target_encoder(model, out)
    unreached = [
        name
        for (name, _), grad in zip(model.target_encoder.named_parameters(), grads)
        if grad is None
    ]
    assert not unreached, f"parameters the probe cannot see even through z: {unreached}"


def test_the_forward_dict_carries_no_state_tensor_that_reaches_the_decoder(
    tiny_kwargs, inputs
):
    """The decoder consumes d_z-wide latents; the only d_model-wide tensors in the output are
    the two encoder states, and neither is connected to the forecasts once z is detached (the
    autograd test above). Here the surface claim: no key named for the removed pathway."""
    model = _model(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(*inputs)
    assert "decoder_state" not in out
    assert model.decoder.proj.body[0].in_features == model.d_z  # first linear reads z, nothing wider
