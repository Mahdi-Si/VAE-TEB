r"""The source pathway never sees the target, and the prior never sees the source.

$\mathrm{KL}(q \Vert p)$ reads as "what the source added" only if $p$, and everything decoded
from $z^p$, is a function of the target's history alone. So replacing the source stream with
noise must leave the prior *and the base forecast* bitwise unchanged -- and, from the other
side, the source encoder must be handed nothing derived from the target.

The bitwise comparisons work because the model is run in ``eval()`` with the generator re-seeded
before each forward: the single ``randn_like`` draw is then the only RNG consumer, so both runs
share their $\epsilon$.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _model(tiny_kwargs) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**tiny_kwargs).eval()


def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(
    tiny_kwargs, inputs
):
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    noise_u = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise_u)

    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "target_state", "z_prior",
                "mu_base", "logvar_base"):
        assert torch.equal(base[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(base["source_state"], resampled["source_state"])


def test_the_source_pathway_receives_only_the_source_stream(tiny_kwargs, inputs):
    """Instrumented at the adapters -- the trust boundary where raw streams enter."""
    model = _model(tiny_kwargs)
    y_st, y_ph, u_stream = inputs
    seen: dict[str, list[torch.Tensor]] = {"source": [], "target": []}

    handles = [
        model.source_adapter.register_forward_pre_hook(
            lambda module, args: seen["source"].append(args[0])
        ),
        model.target_adapter.register_forward_pre_hook(
            lambda module, args: seen["target"].append(args[0])
        ),
    ]
    try:
        with torch.no_grad():
            model(y_st, y_ph, u_stream)
    finally:
        for handle in handles:
            handle.remove()

    # The source adapter got exactly the u_stream object -- not a copy, not a concat that
    # could have mixed a target tensor in.
    assert len(seen["source"]) == 1
    assert seen["source"][0] is u_stream

    # The target adapter got exactly the concatenated target features and nothing source-like:
    # its width is c_y, which cannot hold an extra c_u block.
    assert len(seen["target"]) == 1
    assert torch.equal(seen["target"][0], torch.cat([y_st, y_ph], dim=-1))
    assert seen["target"][0].shape[-1] == model.c_y


def test_the_source_encoder_consumes_only_the_source_adapter_output(tiny_kwargs, inputs):
    """One step deeper: what reaches the source encoder is the adapter's projection of
    u_stream, so no target tensor can join between adapter and encoder."""
    model = _model(tiny_kwargs)
    captured: list[torch.Tensor] = []
    adapter_out: list[torch.Tensor] = []

    handles = [
        model.source_adapter.register_forward_hook(
            lambda module, args, output: adapter_out.append(output)
        ),
        model.source_encoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        ),
    ]
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == 1 and len(adapter_out) == 1
    assert captured[0] is adapter_out[0]
