r"""The shared raw decoder: one instance, one input tensor, zero dropout, no local copy.

No decoder class is written in this package -- the sibling's ``BaselineFutureDecoder`` already
takes one shared core, projects a single input tensor, and emits mean and log-variance heads.
What is pinned here is the *composition*: exactly one instance, whose forward signature admits
exactly one tensor (which is what structurally forbids a conditioning bypass), constructed at
zero dropout because it is invoked twice per forward.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, ResidualFutureDecoder
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _model(tiny_kwargs) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**tiny_kwargs)


def test_exactly_one_decoder_instance_exists(tiny_kwargs):
    model = _model(tiny_kwargs)
    decoders = [m for m in model.modules() if isinstance(m, BaselineFutureDecoder)]
    assert len(decoders) == 1
    assert decoders[0] is model.decoder
    assert not any(isinstance(m, ResidualFutureDecoder) for m in model.modules())


def test_the_decoder_forward_accepts_exactly_the_state_and_a_target_only_persistence(tiny_kwargs):
    """Signature introspection, so a later edit cannot quietly add a bypass argument.

    The admitted set is exactly two names. ``decoder_state`` is the latent path. ``persistence`` is
    the target's own value at the anchor -- a tensor this cell's raw-target composition never
    supplies, since :attr:`persistence_weight` is absent unless the decoder was built for it, and
    the decoder refuses a call that disagrees with how it was built. It is admitted here rather
    than forbidden because it carries no source content by construction and the caller hands the
    **same** tensor to both invocations, so the base-minus-full gap stays a pure source readout;
    a third name would be a bypass and is what this pin exists to catch.
    """
    decoder = _model(tiny_kwargs).decoder
    forward = type(decoder).forward
    parameters = list(inspect.signature(forward).parameters.values())
    assert [p.name for p in parameters] == ["self", "decoder_state", "persistence"]
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters
    ), "no *args/**kwargs escape hatch either"
    # Off on this cell, and off means the parameter is not built rather than merely unused -- so
    # the second name cannot carry anything here even if a call site started passing it.
    assert decoder.persistence_weight is None
    assert inspect.signature(forward).parameters["persistence"].default is None


def test_both_invocations_emit_raw_shaped_forecasts(tiny_kwargs, inputs):
    model = _model(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(*inputs)
    t_valid = model.geometry.t_valid
    expected = (inputs[0].shape[0], t_valid, model.horizon, model.raw_per_step)
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert out[key].shape == expected, key


def test_out_channels_counts_raw_samples_per_horizon_token(tiny_kwargs):
    model = _model(tiny_kwargs)
    assert model.decoder.out_channels == model.raw_per_step == 16


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


def test_the_decoder_projects_from_the_latent_width(tiny_kwargs):
    """The decoder's input is z -- d_z wide -- not an encoder state; a d_model-wide input
    would mean some state pathway was wired in."""
    model = _model(tiny_kwargs)
    with torch.no_grad():
        out = model.decoder(torch.randn(2, 3, model.d_z))
    assert out[0].shape == (2, 3, model.horizon, model.raw_per_step)


def test_no_local_decoders_module_exists():
    """The decoder is reused, not forked; a local nets/decoders.py would be the fork."""
    nets_dir = Path(__file__).resolve().parents[1] / "nets"
    assert not (nets_dir / "decoders.py").exists()
