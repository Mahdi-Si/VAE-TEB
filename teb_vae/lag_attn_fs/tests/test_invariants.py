r"""The four structural invariants, measured on this subclass.

They are what make $\mathrm{KL}(q_t \Vert p_t)$ readable as "what the source added":

1. **Source purity.** The prior, and everything decoded from $z^p$, is a function of the target's
   history alone. Replace the source with noise and neither may move.
2. **No decoder bypass.** Gradient reaches the decoder only through $z$. Without this the latent
   could shrink back into a residual code on a target-derived shortcut, and every reading of
   $\mu^p$ as "the fetal state" would be unsupported.
3. **One shared decoder, invoked twice.** The base and full forecasts differ only in which latent
   they were given -- same module, same weights, same dropout -- so their difference is the
   latent's, not two decoders'.
4. **Exact zero KL at initialisation.** The posterior is a zero-initialised residual on the prior
   under one shared $\epsilon$, so at init the source says exactly nothing.

Every one of them is inherited from the raw sibling, whose own suite pins them on its class; this
file runs them on ``SeqVaeLagAttnFs``, which is what turns "the subclass inherits the invariant"
from an argument about class hierarchies into a measurement. Source purity is additionally run at
the production geometry and reach budget, where the target stream is gathered and delayed. The
zero-KL claim runs under ``TINY_KWARGS``, i.e. the constructor defaults ``base_decode='sample'``
and ``posterior_logvar_mode='residual'``; the shipped flag set is covered by the sibling's
init-policy suite and by the production arm below.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import TINY_KWARGS, shipped_gated_kwargs

_TOL = 1e-6


def _model(kwargs, **overrides):
    torch.manual_seed(0)
    return SeqVaeLagAttnFs(**dict(kwargs, **overrides))


def _closed_form_kl(out: dict) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ per step per dimension, from the returned parameters alone.

    Written out rather than taken from the model, so a model whose own KL readout was wrong
    cannot certify itself.

    Args:
        out: A forward return dict.

    Returns:
        The per-step per-dimension KL.
    """
    return 0.5 * (
        out["logvar_prior"]
        - out["logvar_post"]
        + (out["logvar_post"].exp() + (out["mu_post"] - out["mu_prior"]) ** 2)
        / out["logvar_prior"].exp()
        - 1.0
    )


# ---------------------------------------------------------------------------------------
# 1. Source purity
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["tiny", "shipped-gated"])
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(arm):
    """Bitwise: the model runs in ``eval()`` with the generator re-seeded before each forward, so
    the single ``randn_like`` draw is the only RNG consumer and both runs share their
    $\\epsilon$. The shipped arm gathers and delays the target stream under the reach budget, and
    purity has to survive that."""
    kwargs = dict(TINY_KWARGS) if arm == "tiny" else shipped_gated_kwargs()
    model = _model(kwargs).eval()
    generator = torch.Generator().manual_seed(0)
    length = kwargs["sequence_length"]
    y_st = torch.randn(2, length, 43, generator=generator)
    y_ph = torch.randn(2, length, 66, generator=generator)
    u_stream = torch.randn(2, length, kwargs["c_u"], generator=generator)

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, torch.randn(u_stream.shape, generator=generator))

    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "target_state", "z_prior",
                "mu_base", "logvar_base"):
        assert torch.equal(base[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(base["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(base).abs().max()) == 0.0


# ---------------------------------------------------------------------------------------
# 2. No decoder bypass
# ---------------------------------------------------------------------------------------
def _grads_to_target_encoder(model, out):
    return torch.autograd.grad(
        out["mu_base"].sum(), list(model.target_encoder.parameters()), allow_unused=True
    )


def test_with_z_detached_no_gradient_reaches_the_target_encoder(tiny_kwargs, inputs):
    """The detach happens inside the model's real forward -- by wrapping the sampling method --
    so the probe covers the wiring as built, not a hand-assembled call into the decoder."""
    model = _model(tiny_kwargs)
    sample = model._reparameterize_shared
    model._reparameterize_shared = lambda *args: tuple(  # type: ignore[method-assign]
        z.detach() for z in sample(*args)
    )

    grads = _grads_to_target_encoder(model, model(*inputs))
    leaked = [
        name
        for (name, _), grad in zip(model.target_encoder.named_parameters(), grads)
        if grad is not None
    ]
    assert not leaked, f"target-encoder parameters reachable around z: {leaked}"


def test_without_the_detach_the_same_probe_finds_gradients(tiny_kwargs, inputs):
    """The positive direction: through $z$, every target-encoder parameter is on the path."""
    model = _model(tiny_kwargs)

    grads = _grads_to_target_encoder(model, model(*inputs))
    unreached = [
        name
        for (name, _), grad in zip(model.target_encoder.named_parameters(), grads)
        if grad is None
    ]
    assert not unreached, f"parameters the probe cannot see even through z: {unreached}"


# ---------------------------------------------------------------------------------------
# 3. One shared decoder, invoked twice
# ---------------------------------------------------------------------------------------
def test_the_decoder_is_one_module_invoked_twice(tiny_kwargs, inputs):
    """Counted at the module, not argued from the source: two calls per forward, each taking one
    tensor, into the same object. Two decoders -- or one decoder given a second argument -- would
    make the base-minus-full gap a comparison of two functions rather than of two latents."""
    model = _model(tiny_kwargs).eval()
    calls: list = []

    handle = model.decoder.register_forward_pre_hook(
        lambda module, args: calls.append((module, args))
    )
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        handle.remove()

    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is model.decoder
    assert all(len(args) == 1 for _module, args in calls)


# ---------------------------------------------------------------------------------------
# 4. Exact zero KL at initialisation
# ---------------------------------------------------------------------------------------
def _train_mode_forward(kwargs, inputs, perturb=None):
    """A forward in **train** mode with dropout on, which is where the identities must hold."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(kwargs, dropout=0.1))
    if perturb is not None:
        perturb(model)
    model.train()
    torch.manual_seed(0)
    return model(*inputs)


def test_at_init_the_kl_is_zero_and_both_forecasts_coincide_in_train_mode(tiny_kwargs, inputs):
    """Under ``TINY_KWARGS`` (``base_decode='sample'``, ``posterior_logvar_mode='residual'``).
    Train mode with dropout on is the point: one module, two invocations, and the forecasts are
    bitwise equal only if the decoder carries no dropout, which is what turns the base-minus-full
    readout into a noise-free null."""
    out = _train_mode_forward(tiny_kwargs, inputs)

    assert float(_closed_form_kl(out).abs().max()) == 0.0
    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert float(out["source_kl_lag_map"].abs().max()) == 0.0
    assert torch.equal(out["z_prior"], out["z_post"])
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


def test_everything_above_becomes_false_once_perturbed(tiny_kwargs, inputs, perturb_posterior):
    """The zero must be a property of the init, not of the model being unable to produce a KL.
    Without this, a model whose KL was structurally stuck at zero -- a broken posterior, a
    detached graph -- would pass every test above."""
    out = _train_mode_forward(tiny_kwargs, inputs, perturb=perturb_posterior)

    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert not torch.equal(out["z_prior"], out["z_post"])
    assert not torch.equal(out["mu_base"], out["mu_full"])
