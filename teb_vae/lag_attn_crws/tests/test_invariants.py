r"""The structural invariants, without which the readout means nothing.

Four properties make $\mathrm{KL}(q_t \Vert p_t)$ readable as "what the source added":

1. **Source purity.** The prior, and everything decoded from $z^p$, is a function of the target
   stream's history alone. Replace the source with noise and neither may move; and the traffic runs
   one way, so replacing the target stream must leave the source state untouched.
2. **No decoder bypass.** Gradient reaches the decoder only through $z$. Without it the latent could
   shrink into a residual code on a target-derived shortcut, and reading $\mu^p$ as the predictive
   state would be unsupported.
3. **One shared decoder, invoked twice.** Base and full differ only in which latent they were given
   -- same module, same weights, same (absent) dropout.
4. **Exact zero KL at initialisation**, which lives in ``test_zero_kl_init.py``.

All four are inherited. What is *not* inherited is the decode: this cell gathers the latents at a
tiled anchor set before invoking the decoder, and a gather is where a batch axis and an anchor axis
could be transposed, or where the two branches could be handed different rows. So each test below
runs against **both** classes wherever the fixture allows -- this model and the raw-signal
architecture it claims to inherit from -- which turns "the subclass inherits the invariant" from an
argument about class hierarchies into a measurement.

The architecture is the right comparison arm here and is not available to the causal-feature cells:
those emit $C_{\mathrm{keep}}$ channels per horizon token against its $R$, so their weights diverge
from the first differently-shaped head onward. This cell's do not.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import (
    BATCH,
    TINY_STRIDE,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: Both classes, so every invariant is asserted on this model *and* on the one it inherits it from.
#: The architecture takes no anchor arguments, so the call sites splat a per-class argument list.
_CLASSES = (SeqVaeLagAttnRws, SeqVaeLagAttnCrws)
_CLASS_IDS = ("rws", "crws")


def _kwargs_for(cls) -> dict:
    """The tiny guarded keyword set, with this domain's keywords removed for the architecture."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    if cls is SeqVaeLagAttnCrws:
        return kwargs
    return {
        name: value
        for name, value in kwargs.items()
        if name not in ("target_warmup_steps", "source_warmup_steps", "anchor_stride")
    }


def _extra_args(cls) -> tuple:
    """The anchor arguments, which only this cell's forward takes."""
    return (1, TINY_STRIDE) if cls is SeqVaeLagAttnCrws else ()


def _model(cls, **overrides):
    torch.manual_seed(0)
    return cls(**dict(_kwargs_for(cls), **overrides))


def _closed_form_kl(out: dict) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ from the returned parameters alone, so no model certifies itself."""
    return 0.5 * (
        out["logvar_prior"]
        - out["logvar_post"]
        + (out["logvar_post"].exp() + (out["mu_post"] - out["mu_prior"]) ** 2)
        / out["logvar_prior"].exp()
        - 1.0
    )


# =================================================================================================
# 1. Source purity
# =================================================================================================
@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(cls) -> None:
    """Bitwise: the model runs in ``eval()`` with the generator re-seeded before each forward, so
    the single ``randn_like`` draw is the only RNG consumer and both runs share their $\\epsilon$."""
    model = _model(cls).eval()
    y_st, y_ph, u_stream = make_streams(_kwargs_for(cls))
    extra = _extra_args(cls)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, *extra)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise, *extra)

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


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_resampling_the_target_stream_leaves_the_source_state_unchanged(cls) -> None:
    """The other direction, which the test above cannot give: the traffic is one-way.

    A source encoder that had somehow been handed the target stream would still satisfy every
    "resample the source and the prior does not move" assertion, because that says nothing about
    what the *source* reads.
    """
    model = _model(cls).eval()
    y_st, y_ph, u_stream = make_streams(_kwargs_for(cls))
    extra = _extra_args(cls)
    noise = torch.randn(y_ph.shape, generator=torch.Generator().manual_seed(17))

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, *extra)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, noise, u_stream, *extra)

    assert torch.equal(reference["source_state"], resampled["source_state"])
    assert not torch.equal(reference["target_state"], resampled["target_state"])


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_source_pathway_receives_only_the_source_stream(cls) -> None:
    """Instrumented at the adapters -- the trust boundary where the streams enter."""
    model = _model(cls).eval()
    y_st, y_ph, u_stream = make_streams(_kwargs_for(cls))
    seen: dict = {"source": [], "target": []}

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
            model(y_st, y_ph, u_stream, *_extra_args(cls))
    finally:
        for handle in handles:
            handle.remove()

    assert len(seen["source"]) == 1 and len(seen["target"]) == 1
    # Post-gate on both sides: this family always gathers, on both the guarded and the ungated arm
    # of the architecture's own keyword set.
    assert seen["source"][0].shape[-1] == model.source_gate.out_channels
    assert seen["target"][0].shape[-1] == model.target_gate.out_channels


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_attention_query_is_the_prior_belief(cls) -> None:
    r"""$\mu^p$, and therefore target-only: the lag attention asks a question posed from what the
    target's history alone believes, and the source only answers it.

    Read at the projection's own input rather than argued from the forward's text, and asserted
    against the returned ``mu_prior`` so a query built from some other target-derived tensor -- the
    encoder state, say -- would fail rather than pass as "target-only".
    """
    model = _model(cls).eval()
    assert model.query_uses_logvar is False, "the shipped query is mu^p alone"

    seen: list = []
    handle = model.query_proj.register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(*make_streams(_kwargs_for(cls)), *_extra_args(cls))
    finally:
        handle.remove()

    assert len(seen) == 1
    assert torch.equal(seen[0], out["mu_prior"])


def test_the_forward_takes_no_target() -> None:
    """The forward takes three streams and two integers; it is ``compute_loss`` that gathers a raw
    window afterwards, from a signal the caller passes separately.

    Weaker here than on the feature-target cells, where the reconstruction target is the same *kind*
    of tensor the model is shown -- and worth asserting anyway, because the raw signal is a batch
    field this model could be handed and must not be.
    """
    parameters = list(inspect.signature(SeqVaeLagAttnCrws.forward).parameters)
    assert parameters == ["self", "y_st", "y_ph", "u_stream", "anchor_phase", "anchor_stride"]

    model = _model(SeqVaeLagAttnCrws).eval()
    with torch.no_grad():
        out = model(*make_streams(_kwargs_for(SeqVaeLagAttnCrws)), 1, TINY_STRIDE)

    # The one key whose name contains "target" is the encoder's history state, at d_model rather
    # than at the decoder's width -- so no returned tensor could be a forecast target.
    assert [key for key in out if "target" in key] == ["target_state"]
    assert out["target_state"].shape[-1] == model.d_model != model.decoder_out_channels


# =================================================================================================
# 2. No decoder bypass
# =================================================================================================
def _grads_to_target_encoder(model, out):
    return torch.autograd.grad(
        out["mu_base"].sum(), list(model.target_encoder.parameters()), allow_unused=True
    )


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_with_z_detached_no_gradient_reaches_the_target_encoder(cls) -> None:
    """The detach happens inside the model's real forward -- by wrapping the sampling method -- so
    the probe covers the wiring as built, not a hand-assembled call into the decoder."""
    model = _model(cls)
    sample = model._reparameterize_shared
    model._reparameterize_shared = lambda *args: tuple(  # type: ignore[method-assign]
        z.detach() for z in sample(*args)
    )

    out = model(*make_streams(_kwargs_for(cls)), *_extra_args(cls))
    leaked = [
        name
        for (name, _), grad in zip(
            model.target_encoder.named_parameters(), _grads_to_target_encoder(model, out)
        )
        if grad is not None
    ]
    assert not leaked, f"target-encoder parameters reachable around z: {leaked}"


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_without_the_detach_the_same_probe_finds_gradients(cls) -> None:
    """The positive direction: through $z$, every target-encoder parameter is on the path.

    On this model that is a statement about the *gather* as well: a decode at anchors none of which
    the encoder reaches would leave parameters unreached and look exactly like a bypass.
    """
    model = _model(cls)
    out = model(*make_streams(_kwargs_for(cls)), *_extra_args(cls))
    unreached = [
        name
        for (name, _), grad in zip(
            model.target_encoder.named_parameters(), _grads_to_target_encoder(model, out)
        )
        if grad is None
    ]
    assert not unreached, f"parameters the probe cannot see even through z: {unreached}"


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_reads_the_latent_and_nothing_wider(cls) -> None:
    """The surface claim beside the autograd one: the decoder's first linear reads $d_z$."""
    model = _model(cls)

    assert model.decoder.proj.body[0].in_features == model.d_z


# =================================================================================================
# 3. One shared decoder, invoked twice
# =================================================================================================
@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_is_one_module_invoked_twice(cls) -> None:
    """Counted at the module: two calls per forward, each taking one tensor, into the same object.

    And on this model the two calls must receive the **same** anchor rows -- two gathers at two
    indices would make the base-minus-full gap a comparison of two anchor sets, and therefore of two
    different raw windows.
    """
    model = _model(cls).eval()
    calls: list = []
    handle = model.decoder.register_forward_pre_hook(
        lambda module, args: calls.append((module, args))
    )
    try:
        with torch.no_grad():
            model(*make_streams(_kwargs_for(cls)), *_extra_args(cls))
    finally:
        handle.remove()

    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is model.decoder
    assert all(len(args) == 1 for _module, args in calls)
    assert calls[0][1][0].shape == calls[1][1][0].shape
    assert not any(hasattr(model, name) for name in ("residual_decoder", "baseline_decoder"))


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_carries_no_dropout(cls) -> None:
    """What makes the two invocations comparable in train mode: two dropout masks would put noise
    into every base-minus-full readout, and the gap would be reported as coupling."""
    model = _model(cls, dropout=0.1)

    dropouts = [
        module.p for module in model.decoder.modules() if isinstance(module, torch.nn.Dropout)
    ]
    assert all(probability == 0.0 for probability in dropouts), dropouts
    assert model.lag_attn.attn_dropout.p == 0.0


# =================================================================================================
# The production geometry and budget
# =================================================================================================
def test_the_invariants_hold_at_the_production_geometry_and_budget() -> None:
    """One pass at the real thing: $300$ steps, the budget the committed shard resolves, and eleven
    tiles of a one-minute raw block.

    The tiny fixture's guard is hand-built; this one is resolved from the shard, so the invariants
    are asserted against the geometry a run would actually train at.
    """
    kwargs = shipped_warmup_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**dict(kwargs, dropout=0.0)).eval()
    y_st, y_ph, u_stream = make_streams(kwargs, batch=BATCH)
    phase = torch.tensor([0, 7])

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, phase)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(
            y_st,
            y_ph,
            torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(7)),
            phase,
        )

    for key in ("mu_prior", "logvar_prior", "z_prior", "mu_base", "logvar_base"):
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) == 0.0
    assert tuple(reference["mu_base"].shape) == (BATCH, 5, 30, 16)
