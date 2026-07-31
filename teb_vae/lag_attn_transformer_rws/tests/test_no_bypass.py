r"""Gradient reaches the decoder only through $z$ -- the claim made operational.

"No decoder bypass" is what forces the prior latent to carry the target's predictive state: if any
target-derived tensor reached the decoder around the latent, $z$ could shrink back into a residual
code and every downstream reading of $\mu^p$ as "the fetal state" would be unsupported. It is also
what makes the source-conditioned KL comparable between this architecture and the one it replaces:
both must route everything through a latent of the same width.

Rhetoric is cheap here, so the test is autograd: with the latent samples detached inside the
model's own forward, no gradient path from either forecast to either encoder may survive. Both
directions are asserted -- the same probe without the detach must find gradients -- so the test
cannot pass vacuously on a broken graph.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws


def _model(tiny_kwargs) -> SeqVaeLagAttnTrfRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfRws(**tiny_kwargs)


def _detach_latents(model: SeqVaeLagAttnTrfRws) -> None:
    """Detach both latent samples inside the model's real forward, by wrapping the sampler.

    Wrapped rather than hand-assembled: the probe then covers the wiring as built, including the
    two decoder invocations, rather than a call into the decoder written by the test.
    """
    sample = model._reparameterize_shared
    model._reparameterize_shared = lambda *args: tuple(  # type: ignore[method-assign]
        z.detach() for z in sample(*args)
    )


def _encoder_grads(model: SeqVaeLagAttnTrfRws, tensor: torch.Tensor, encoder: str):
    """Gradients of ``tensor.sum()` with respect to one encoder's parameters, ``None`` allowed."""
    parameters = list(getattr(model, encoder).parameters())
    return parameters, torch.autograd.grad(
        tensor.sum(), parameters, retain_graph=True, allow_unused=True
    )


@pytest.mark.parametrize("forecast", ["mu_base", "mu_full"])
@pytest.mark.parametrize("encoder", ["target_encoder", "source_encoder"])
def test_with_z_detached_no_gradient_reaches_either_encoder(
    tiny_kwargs, inputs, forecast, encoder
):
    model = _model(tiny_kwargs)
    _detach_latents(model)

    out = model(*inputs)
    parameters, grads = _encoder_grads(model, out[forecast], encoder)
    named = list(getattr(model, encoder).named_parameters())
    leaked = [name for (name, _), grad in zip(named, grads) if grad is not None]

    assert parameters, f"{encoder} has no parameters; the probe is vacuous"
    assert not leaked, f"{encoder} parameters reachable from {forecast} around z: {leaked}"


def test_without_the_detach_the_same_probe_finds_gradients(tiny_kwargs, inputs):
    """The positive direction: through $z$, every target-encoder parameter is on the path."""
    model = _model(tiny_kwargs)
    out = model(*inputs)

    _, grads = _encoder_grads(model, out["mu_base"], "target_encoder")
    named = list(model.target_encoder.named_parameters())
    unreached = [name for (name, _), grad in zip(named, grads) if grad is None]

    assert not unreached, f"parameters the probe cannot see even through z: {unreached}"


def test_the_full_forecast_reaches_the_source_encoder_through_the_latent(tiny_kwargs, inputs):
    """The other half of the positive direction, which the base forecast cannot show: the source
    is only ever read by the posterior, so ``mu_full`` is where its gradient has to appear."""
    model = _model(tiny_kwargs)
    out = model(*inputs)

    _, grads = _encoder_grads(model, out["mu_full"], "source_encoder")
    named = list(model.source_encoder.named_parameters())
    unreached = [name for (name, _), grad in zip(named, grads) if grad is None]

    assert not unreached, f"the source encoder is unreachable even through z: {unreached}"


def test_the_decoder_reads_a_latent_and_nothing_wider(tiny_kwargs, inputs):
    """The decoder consumes $d_z$-wide latents; the only $d_{model}$-wide tensors in the output are
    the two encoder states, and neither is connected to the forecasts once $z$ is detached. Here
    the surface claim: the first linear reads $z$, and no key is named for the removed pathway."""
    model = _model(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(*inputs)

    assert "decoder_state" not in out
    assert model.decoder.proj.body[0].in_features == model.d_z
