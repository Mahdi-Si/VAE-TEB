r"""The four structural invariants, without which the readout means nothing.

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

Every one of them is inherited: this model changes what the decoder emits, not what feeds it. So
each test runs against **both** classes wherever the fixture allows, which is what turns "the
subclass inherits the invariant" from an argument about class hierarchies into a measurement. A
divergence would show as one parametrisation failing.

The fourth carries a caveat that has to be stated rather than assumed. The zero-KL claim is
asserted under ``TINY_KWARGS``, which sets neither ``base_decode`` nor ``posterior_logvar_mode``
and so runs the constructor defaults ``'sample'`` and ``'residual'``. The shipped configuration
sets ``base_decode: mean``, under which the two forecasts are *not* bitwise identical at init
because only $z^q$ is sampled, and ``posterior_logvar_mode: independent``, under which the KL is
zero at init only with ``head_init_calibration: true``. Both shipped settings are covered
separately at the bottom of this file rather than left to the reader.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import shipped_gated_kwargs
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

#: Both classes, so every invariant below is asserted on the subclass *and* on the model it
#: claims to inherit it from.
_CLASSES = (SeqVaeLagAttnRws, SeqVaeLagAttnFs)
_CLASS_IDS = ("rws", "fs")

_TOL = 1e-6


def _model(cls, kwargs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


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
@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(
    cls, tiny_kwargs, inputs
):
    """Bitwise: the model runs in ``eval()`` with the generator re-seeded before each forward, so
    the single ``randn_like`` draw is the only RNG consumer and both runs share their
    $\\epsilon$."""
    model = _model(cls, tiny_kwargs).eval()
    y_st, y_ph, u_stream = inputs

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise)

    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "target_state", "z_prior",
                "mu_base", "logvar_base"):
        assert torch.equal(base[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(base["source_state"], resampled["source_state"])


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_source_pathway_receives_only_the_source_stream(cls, tiny_kwargs, inputs):
    """Instrumented at the adapters -- the trust boundary where the streams enter."""
    model = _model(cls, tiny_kwargs).eval()
    y_st, y_ph, u_stream = inputs
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
            model(y_st, y_ph, u_stream)
    finally:
        for handle in handles:
            handle.remove()

    assert len(seen["source"]) == 1 and seen["source"][0] is u_stream
    assert len(seen["target"]) == 1
    assert torch.equal(seen["target"][0], torch.cat([y_st, y_ph], dim=-1))
    assert seen["target"][0].shape[-1] == model.c_y


def test_the_forward_takes_no_target(tiny_kwargs, inputs):
    """The invariant this target domain adds. The reconstruction target is now built from the
    *same kind of tensor the model is shown*, so the two could be confused in a way the raw
    sibling's cannot -- and a forward that had somehow been handed the future would still return
    every contract shape. The forward takes three streams and holds no target: it is
    ``compute_loss`` that gathers one, after the forward has run, from a stream the caller
    passes it separately. Asserted on the signature, which is where the confusion would have to
    enter."""
    import inspect

    parameters = list(inspect.signature(SeqVaeLagAttnFs.forward).parameters)
    assert parameters == ["self", "y_st", "y_ph", "u_stream"]

    model = _model(SeqVaeLagAttnFs, tiny_kwargs).eval()
    with torch.no_grad():
        out = model(*inputs)
    # The one key whose name contains "target" is the encoder's history state, at d_model rather
    # than at the decoder's width -- so no returned tensor could be a forecast target.
    assert [key for key in out if "target" in key] == ["target_state"]
    assert out["target_state"].shape[-1] == model.d_model != model.decoder_out_channels


# ---------------------------------------------------------------------------------------
# 2. No decoder bypass
# ---------------------------------------------------------------------------------------
def _grads_to_target_encoder(model, out):
    return torch.autograd.grad(
        out["mu_base"].sum(), list(model.target_encoder.parameters()), allow_unused=True
    )


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_with_z_detached_no_gradient_reaches_the_target_encoder(cls, tiny_kwargs, inputs):
    """The detach happens inside the model's real forward -- by wrapping the sampling method --
    so the probe covers the wiring as built, not a hand-assembled call into the decoder."""
    model = _model(cls, tiny_kwargs)
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


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_without_the_detach_the_same_probe_finds_gradients(cls, tiny_kwargs, inputs):
    """The positive direction: through $z$, every target-encoder parameter is on the path."""
    model = _model(cls, tiny_kwargs)

    grads = _grads_to_target_encoder(model, model(*inputs))
    unreached = [
        name
        for (name, _), grad in zip(model.target_encoder.named_parameters(), grads)
        if grad is None
    ]
    assert not unreached, f"parameters the probe cannot see even through z: {unreached}"


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_reads_the_latent_and_nothing_wider(cls, tiny_kwargs):
    """The surface claim beside the autograd one: the decoder's first linear reads $d_z$, so no
    $d_{\\mathrm{model}}$-wide encoder state could be fed to it even if one were passed."""
    model = _model(cls, tiny_kwargs)

    assert model.decoder.proj.body[0].in_features == model.d_z


# ---------------------------------------------------------------------------------------
# 3. One shared decoder, invoked twice
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_is_one_module_invoked_twice(cls, tiny_kwargs, inputs):
    """Counted at the module, not argued from the source: two calls per forward, each taking one
    tensor, into the same object. Two decoders -- or one decoder given a second argument -- would
    make the base-minus-full gap a comparison of two functions rather than of two latents."""
    model = _model(cls, tiny_kwargs).eval()
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
    # The two invocations differ in exactly one thing: which latent they were handed.
    assert not any(hasattr(model, name) for name in ("residual_decoder", "baseline_decoder"))


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_carries_no_dropout(cls, tiny_kwargs):
    """What makes the two invocations comparable in train mode: two dropout masks would put noise
    into every base-minus-full readout, and the gap would be reported as coupling."""
    model = _model(cls, tiny_kwargs, dropout=0.1)

    dropouts = [
        module.p
        for module in model.decoder.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    assert all(probability == 0.0 for probability in dropouts), dropouts
    assert model.lag_attn.attn_dropout.p == 0.0


# ---------------------------------------------------------------------------------------
# 4. Exact zero KL at initialisation
# ---------------------------------------------------------------------------------------
def _train_mode_forward(cls, kwargs, inputs, perturb=None):
    """A forward in **train** mode with dropout on, which is where the identities must hold."""
    torch.manual_seed(0)
    model = cls(**dict(kwargs, dropout=0.1))
    if perturb is not None:
        perturb(model)
    model.train()
    torch.manual_seed(0)
    return model(*inputs)


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_kl_is_exactly_zero_at_init(cls, tiny_kwargs, inputs):
    """Under ``TINY_KWARGS``: ``base_decode='sample'`` and ``posterior_logvar_mode='residual'``,
    the constructor defaults, neither of which the tiny keyword set names."""
    assert "base_decode" not in tiny_kwargs and "posterior_logvar_mode" not in tiny_kwargs
    out = _train_mode_forward(cls, tiny_kwargs, inputs)

    assert float(_closed_form_kl(out).abs().max()) == 0.0
    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert float(out["source_kl_lag_map"].abs().max()) == 0.0


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_two_forecasts_are_bitwise_identical_at_init_in_train_mode(cls, tiny_kwargs, inputs):
    """One module, two invocations, two dropout masks -- unless the decoder has no dropout, which
    is what makes this exact. Train mode is the point: it is what turns the base-minus-full
    readout into a noise-free null."""
    out = _train_mode_forward(cls, tiny_kwargs, inputs)

    assert torch.equal(out["z_prior"], out["z_post"])
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_everything_above_becomes_false_once_perturbed(
    cls, tiny_kwargs, inputs, perturb_posterior
):
    """The zero must be a property of the init, not of the model being unable to produce a KL.
    Without this, a model whose KL was structurally stuck at zero -- a broken posterior, a
    detached graph -- would pass every test above."""
    out = _train_mode_forward(cls, tiny_kwargs, inputs, perturb=perturb_posterior)

    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert not torch.equal(out["z_prior"], out["z_post"])
    assert not torch.equal(out["mu_base"], out["mu_full"])


# ---------------------------------------------------------------------------------------
# The shipped flags, which the tiny fixture does not set
# ---------------------------------------------------------------------------------------
def test_under_the_shipped_posterior_mode_the_init_kl_is_still_exactly_zero(
    tiny_kwargs, inputs
):
    """``posterior_logvar_mode: independent`` writes the posterior log-variance from its own head
    rather than as a residual, so the zero is bought by ``head_init_calibration: true`` -- which
    puts that head and the prior's on the same value -- rather than by the residual being zero.
    The shipped configuration sets both, and the pairing is what makes the KL start at zero."""
    out = _train_mode_forward(
        SeqVaeLagAttnFs,
        dict(tiny_kwargs, posterior_logvar_mode="independent", head_init_calibration=True),
        inputs,
    )

    assert float(_closed_form_kl(out).abs().max()) == 0.0
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


def test_under_the_shipped_base_decode_the_forecasts_are_not_bitwise_identical(
    tiny_kwargs, inputs
):
    """``base_decode: mean`` decodes the base branch from $\\mu^p$ rather than from a draw, so at
    init the two forecasts differ by exactly the sampling noise $z^q$ carries. Stated here because
    a reader who took the bitwise identity above as unconditional would read the shipped run's
    non-zero init ``pred_gap`` as a bug."""
    out = _train_mode_forward(SeqVaeLagAttnFs, dict(tiny_kwargs, base_decode="mean"), inputs)

    assert float(_closed_form_kl(out).abs().max()) == 0.0  # the KL is still exactly zero
    assert torch.equal(out["z_prior"], out["mu_prior"])
    assert not torch.equal(out["mu_base"], out["mu_full"])


def test_the_invariants_hold_at_the_production_geometry_and_budget():
    """One pass at the real thing. The tiny fixture is ungated; the shipped one gathers 78 of 109
    target channels and delays every survivor, and source purity has to survive that."""
    kwargs = shipped_gated_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(kwargs, dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(0)
    length = kwargs["sequence_length"]
    y_st = torch.randn(2, length, 43, generator=generator)
    y_ph = torch.randn(2, length, 66, generator=generator)
    u_stream = torch.randn(2, length, 58, generator=generator)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, torch.randn(u_stream.shape, generator=generator))

    for key in ("mu_prior", "logvar_prior", "z_prior", "mu_base", "logvar_base"):
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) == 0.0
