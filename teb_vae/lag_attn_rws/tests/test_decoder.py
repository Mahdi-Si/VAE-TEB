r"""The shared raw decoder: one instance, zero dropout.

No decoder class is written in this package -- the sibling's ``BaselineFutureDecoder`` already
takes one shared core, projects a single input tensor, and emits mean and log-variance heads.
What is pinned here is the *composition*: exactly one instance, so the base and full branches
decode through the same weights, constructed at zero dropout because it is invoked twice per
forward. The forecast shapes are asserted by the forward-contract and construction suites.
"""
from __future__ import annotations

import torch
from torch import nn

from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, ResidualFutureDecoder
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def test_exactly_one_decoder_instance_exists(tiny_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**tiny_kwargs)
    decoders = [m for m in model.modules() if isinstance(m, BaselineFutureDecoder)]
    assert len(decoders) == 1
    assert decoders[0] is model.decoder
    assert not any(isinstance(m, ResidualFutureDecoder) for m in model.modules())


def test_decoder_dropout_is_zero(tiny_kwargs):
    """Invoked twice per forward: two independent dropout masks would make base and full
    differ at init and inject noise into the base-minus-full readout on every step. Even with
    encoder dropout on, the decoder subtree must carry none."""
    model = SeqVaeLagAttnRws(**dict(tiny_kwargs, dropout=0.1))
    offenders = [
        name
        for name, module in model.decoder.named_modules()
        if isinstance(module, nn.Dropout) and module.p > 0.0
    ]
    assert not offenders, f"dropout inside the shared decoder: {offenders}"
