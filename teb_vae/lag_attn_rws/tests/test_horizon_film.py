r"""Per-block FiLM in the shared decoder, with a true identity at initialisation.

The horizon core re-injects the latent at *every* refine block, not once at the top of the stack:
otherwise the 30 horizon tokens enter the refine stack ~97.6% identical and the core can synthesise
the trajectory shape in a direction the latent does not control. Reading $z$ at every block
forecloses that.

The identity-at-init is the subtle part. The core zero-inits its FiLM generators, but the model's
generic ``initialization`` xavier-refills every ``nn.Linear`` afterwards -- so without a re-zero the
"identity FiLM at init" the docstrings promise is silently false (the shipped single-FiLM sibling
runs FiLM *random* at init). The model re-zeros the generators in its post-init block,
unconditionally, so at step 0 the per-block-FiLM decoder is bitwise the FiLM-free decoder and the
latent enters the trajectory only as training moves the generators off zero.
"""
from __future__ import annotations

import torch
from torch import nn

from teb_vae.lag_attn.nets.decoders import HorizonDecoderCore
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _model(tiny_kwargs) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**tiny_kwargs)


def test_the_core_uses_per_block_film_not_a_single_generator(tiny_kwargs):
    """Hardcoded per-block FiLM: the generators live inside the refine stack and the single
    top-of-stack ``film_gen`` is not built, so no dead parameter sits in DDP's expectation set."""
    core = _model(tiny_kwargs).horizon_core
    assert core.film_per_block is True
    assert core.film_gen is None
    assert core.refine.film is not None
    assert len(core.refine.film) == len(core.refine.blocks)


def test_every_film_generator_is_exactly_zero_after_construction(tiny_kwargs):
    """The R3 re-zero: ``initialization`` xavier-fills the generators, and the post-init re-zero
    must put them back to exactly zero -- weights *and* biases -- or the identity-at-init is a
    fiction."""
    core = _model(tiny_kwargs).horizon_core
    assert core.refine.film is not None
    for layer in core.refine.film:
        assert isinstance(layer, nn.Linear)
        assert layer.weight.abs().max().item() == 0.0, "FiLM generator weight not re-zeroed"
        assert layer.bias.abs().max().item() == 0.0, "FiLM generator bias not re-zeroed"


def test_the_init_decode_bitwise_equals_a_film_free_pass(tiny_kwargs):
    """The identity made concrete: with the generators zeroed, the per-block-FiLM core's decode is
    ``torch.equal`` to a FiLM-free core holding the same shared weights. If ``initialization`` had
    been allowed to leave FiLM random, this would fail."""
    core = _model(tiny_kwargs).horizon_core
    # Everything except FiLM is mirrored off the built core, the horizon attention included: this
    # is a FiLM comparison, and a reference that silently dropped the attention blocks would make
    # it a comparison of two different decoders the moment the shipped config turns them on.
    reference = HorizonDecoderCore(
        d_hidden=core.d_hidden,
        horizon=core.horizon,
        depth=len(core.refine.blocks),
        film=False,
        attention_blocks=core.attention_blocks,
        attention_heads=core.attention_heads,
    )
    # Copy the shared (non-FiLM) weights; the FiLM generators have no counterpart and are dropped.
    reference.load_state_dict(core.state_dict(), strict=False)

    h = torch.randn(3, 5, core.d_hidden)
    with torch.no_grad():
        assert torch.equal(core.decode(h), reference.decode(h))


def test_gradient_reaches_the_latent_through_every_per_block_film_generator(tiny_kwargs):
    """Connectivity, deterministically and without an optimiser loop: with the generators
    perturbed off zero, one backward from the decode output gives the latent input a gradient and
    puts every per-block generator on the graph."""
    core = _model(tiny_kwargs).horizon_core
    assert core.refine.film is not None
    generators = [layer for layer in core.refine.film]

    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in generators:
            assert isinstance(layer, nn.Linear)
            layer.weight.add_(torch.randn(layer.weight.shape, generator=generator) * 0.1)

    h = torch.randn(2, 3, core.d_hidden, requires_grad=True)
    core.decode(h).pow(2).sum().backward()

    # h is the projected latent -- the only entry point z has into the decoder.
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0
    for index, layer in enumerate(generators):
        assert isinstance(layer, nn.Linear)
        assert layer.weight.grad is not None and float(layer.weight.grad.abs().sum()) > 0.0, (
            f"per-block FiLM generator {index} received no gradient"
        )


def test_the_film_branch_itself_routes_gradient_to_the_latent(tiny_kwargs):
    """Stronger than 'z is connected': the FiLM path *specifically* carries gradient to the latent.
    At zero the multiplicative branch is an identity and contributes nothing to $\\partial/\\partial
    h$; perturbed, it changes the gradient reaching $h$ -- so the latent is genuinely modulated
    through the per-block generators, not only through the convolutions."""
    core = _model(tiny_kwargs).horizon_core
    assert core.refine.film is not None

    def latent_grad() -> torch.Tensor:
        base = torch.randn(2, 3, core.d_hidden, generator=torch.Generator().manual_seed(1))
        h = base.clone().requires_grad_(True)
        core.decode(h).pow(2).sum().backward()
        assert h.grad is not None
        return h.grad.clone()

    grad_identity = latent_grad()

    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for layer in core.refine.film:
            assert isinstance(layer, nn.Linear)
            layer.weight.add_(torch.randn(layer.weight.shape, generator=generator) * 0.3)

    grad_active = latent_grad()
    assert not torch.equal(grad_identity, grad_active), (
        "perturbing the FiLM generators did not change the gradient reaching the latent, so the "
        "FiLM branch is not routing gradient to z"
    )
