r"""The structural invariants, without which the readout means nothing.

Three properties make $\mathrm{KL}(q_t \Vert p_t)$ readable as "what the source added", asserted on
this cell's composed model -- which gathers the latents at a tiled anchor set before decoding, where
a batch axis and an anchor axis could be transposed or the two branches handed different rows:

1. **Source purity.** The prior, and everything decoded from $z^p$, is a function of the target
   stream's history alone. Replace the source with noise and neither may move -- also with the
   prior's availability clock on, because the clock is the encode of an all-zero stream and so a
   function of $t$ and the configuration alone, never of the source's values. The traffic runs one
   way, so replacing the target stream must leave the source state untouched; and the lag
   attention's query is $\mu^p$ itself.
2. **No decoder bypass.** Gradient reaches the target encoder from the base forecast only through
   $z$, and through $z$ it reaches every parameter.
3. **Exact zero KL at initialisation**, which lives in ``test_zero_kl_init.py``.

The same invariants of the raw-signal architecture, and the clock's own properties, are tested by
the cells that own them.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws

from .conftest import (
    TINY_STRIDE,
    make_streams,
    tiny_warmup_kwargs,
)

#: The anchor arguments every forward below takes.
_ANCHOR_ARGS = (1, TINY_STRIDE)


def _kwargs() -> dict:
    """The tiny guarded keyword set at a real tiling."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


def _model(**overrides) -> SeqVaeLagAttnCrws:
    torch.manual_seed(0)
    return SeqVaeLagAttnCrws(**dict(_kwargs(), **overrides))


# =================================================================================================
# 1. Source purity
# =================================================================================================
@pytest.mark.parametrize("clock", [False, True], ids=["no_clock", "clock"])
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(clock: bool) -> None:
    """Bitwise: the model runs in ``eval()`` with the generator re-seeded before each forward, so
    the single ``randn_like`` draw is the only RNG consumer and both runs share their $\\epsilon$.

    With the clock on the prior takes a second input built from the source pathway; if the clock
    were an encode of the *actual* source rather than of silence, every tensor below would move and
    every shape would be unchanged."""
    model = _model(prior_availability_input=clock).eval()
    y_st, y_ph, u_stream = make_streams(_kwargs())

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, *_ANCHOR_ARGS)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise, *_ANCHOR_ARGS)

    for key in (
        "mu_prior",
        "logvar_prior",
        "raw_logvar_prior",
        "target_state",
        "z_prior",
        "mu_base",
        "logvar_base",
    ):
        assert torch.equal(reference[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(reference["source_state"], resampled["source_state"])


def test_resampling_the_target_stream_leaves_the_source_state_unchanged() -> None:
    """The other direction, which the test above cannot give: the traffic is one-way.

    A source encoder that had somehow been handed the target stream would still satisfy every
    "resample the source and the prior does not move" assertion, because that says nothing about
    what the *source* reads.
    """
    model = _model().eval()
    y_st, y_ph, u_stream = make_streams(_kwargs())
    noise = torch.randn(y_ph.shape, generator=torch.Generator().manual_seed(17))

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, *_ANCHOR_ARGS)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, noise, u_stream, *_ANCHOR_ARGS)

    assert torch.equal(reference["source_state"], resampled["source_state"])
    assert not torch.equal(reference["target_state"], resampled["target_state"])


def test_the_attention_query_is_the_prior_belief() -> None:
    r"""$\mu^p$, and therefore target-only: the lag attention asks a question posed from what the
    target's history alone believes, and the source only answers it.

    Read at the projection's own input rather than argued from the forward's text, and asserted
    against the returned ``mu_prior`` so a query built from some other target-derived tensor -- the
    encoder state, say -- would fail rather than pass as "target-only".
    """
    model = _model().eval()
    assert model.query_uses_logvar is False, "the shipped query is mu^p alone"

    seen: list = []
    handle = model.query_proj.register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(*make_streams(_kwargs()), *_ANCHOR_ARGS)
    finally:
        handle.remove()

    assert len(seen) == 1
    assert torch.equal(seen[0], out["mu_prior"])


# =================================================================================================
# 2. No decoder bypass
# =================================================================================================
def _target_encoder_grads(model) -> dict:
    """Gradient of the summed base forecast w.r.t. each target-encoder parameter (``None`` if
    unreached), through the model's real forward."""
    out = model(*make_streams(_kwargs()), *_ANCHOR_ARGS)
    names, parameters = zip(*model.target_encoder.named_parameters())
    grads = torch.autograd.grad(out["mu_base"].sum(), list(parameters), allow_unused=True)
    return dict(zip(names, grads))


def test_gradient_reaches_the_target_encoder_only_through_z() -> None:
    """The detach happens inside the model's real forward -- by wrapping the sampling method -- so
    the probe covers the wiring as built, not a hand-assembled call into the decoder.

    The positive direction first: through $z$ every target-encoder parameter is on the path. On
    this model that is a statement about the *gather* as well: a decode at anchors none of which
    the encoder reaches would leave parameters unreached and look exactly like a bypass.
    """
    unreached = [name for name, grad in _target_encoder_grads(_model()).items() if grad is None]
    assert not unreached, f"parameters the probe cannot see even through z: {unreached}"

    model = _model()
    sample = model._reparameterize_shared
    model._reparameterize_shared = lambda *args: tuple(  # type: ignore[method-assign]
        z.detach() for z in sample(*args)
    )
    leaked = [name for name, grad in _target_encoder_grads(model).items() if grad is not None]
    assert not leaked, f"target-encoder parameters reachable around z: {leaked}"
