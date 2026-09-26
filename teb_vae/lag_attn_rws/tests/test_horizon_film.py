r"""Per-block FiLM in the shared decoder is a true identity at initialisation, inside this model.

The core zero-inits its FiLM generators, but the model's generic ``initialization`` xavier-refills
every ``nn.Linear`` afterwards -- so without a re-zero the "identity FiLM at init" is silently
false. The model re-zeros the generators in its post-init block, so at step $0$ the
per-block-FiLM decoder is bitwise the FiLM-free decoder. The bare core's own FiLM behaviour
(generators built per block, zero at construction, all on the gradient path) is pinned by
``teb_vae/lag_attn/tests/test_logvar_floor.py``; this file pins only what the assembled model adds.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.nets.decoders import HorizonDecoderCore
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _model(tiny_kwargs) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**tiny_kwargs)


def test_the_init_decode_bitwise_equals_a_film_free_pass(tiny_kwargs):
    """The identity made concrete: with the generators zeroed, the per-block-FiLM core's decode is
    ``torch.equal`` to a FiLM-free core holding the same shared weights. If ``initialization`` had
    been allowed to leave FiLM random, this would fail."""
    core = _model(tiny_kwargs).horizon_core
    assert core.refine.film is not None, "no per-block FiLM was built, so the identity is vacuous"
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
