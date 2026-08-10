r"""The four structural invariants, re-run over this composition rather than restated.

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

**Every definition is imported, not copied.** ``_TARGET_ONLY_KEYS``, ``_capture_adapter_inputs``
and ``CrossWiredModel`` come from
``lag_attn_transformer_rws/tests/test_source_purity.py``; ``_closed_form_kl`` and its tolerance from
``test_zero_kl_init.py``; ``_detach_latents`` and ``_encoder_grads`` from ``test_no_bypass.py``.
What this file supplies is the class. That is the point: an invariant whose definition lived in two
places could hold in one and drift in the other, and every one of the four is inherited here --
this model changes what the decoder emits, not what feeds it.

The negative controls come across with them, which is what stops the assertions being vacuous. Each
is of the form "this pathway did not move", and a model that computed nothing at all would satisfy
every one; ``CrossWiredModel`` is the model that must fail them, composed with the target mixin here
so the control runs against a *feature* forecaster rather than against the raw one it was written
for.

**The fourth invariant's flags have to be stated rather than assumed.** The zero-KL claim is
asserted under the conv-Transformer suite's ``TINY_KWARGS``, which names neither ``base_decode`` nor
``posterior_logvar_mode`` and so runs the constructor defaults ``'sample'`` and ``'residual'``. The
shipped configuration sets ``base_decode: mean``, under which the two forecasts are *not* bitwise
identical at init because only $z^q$ is sampled, and ``posterior_logvar_mode: independent``, under
which the KL is zero at init only with ``head_init_calibration: true``. Both are covered separately
at the bottom rather than left to the reader.

The last section is the one thing here that is genuinely new. ``head_init_calibration`` is the only
initialisation policy the width change reaches, and it now runs against a $78$-wide output head
instead of a $16$-wide one -- so it is measured at that width rather than inherited, and the two
policies that must *not* have moved are measured beside it.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import BATCH, shipped_gated_kwargs
from teb_vae.lag_attn_transformer_rws.tests.test_no_bypass import (
    _detach_latents,
    _encoder_grads,
)
from teb_vae.lag_attn_transformer_rws.tests.test_source_purity import (
    _TARGET_ONLY_KEYS,
    CrossWiredModel,
    _capture_adapter_inputs,
)
from teb_vae.lag_attn_transformer_rws.tests.test_zero_kl_init import _TOL, _closed_form_kl


class _CrossWiredFeatureModel(SeqVaeLagAttnTrfFs, CrossWiredModel):
    """The imported negative control, composed with this target domain.

    Its ``forward`` is ``CrossWiredModel``'s -- the streams are leaked into each other before the
    real forward runs -- reached by putting it after this class in the bases, so the control is the
    sibling's definition applied to a *feature* forecaster rather than a second cross-wiring
    written here.
    """


def _model(kwargs, cls=SeqVaeLagAttnTrfFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _train_mode_forward(kwargs, inputs, perturb=None, cls=SeqVaeLagAttnTrfFs):
    """A forward in **train** mode with dropout on, which is where the identities must hold.

    Dropout deliberately on: both the prior and the posterior are read off *one* common encoder
    forward, so whatever activations dropout removes it removes from both branches identically. Two
    encoder passes would break every identity below even with the deltas at zero.
    """
    torch.manual_seed(0)
    model = cls(**dict(kwargs, dropout=0.1))
    if perturb is not None:
        perturb(model)
    model.train()
    torch.manual_seed(0)
    return model(*inputs)


# ---------------------------------------------------------------------------------------
# 1. Source purity
# ---------------------------------------------------------------------------------------
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged(tiny_gated, inputs):
    """The seven target-only keys are the sibling's list, imported: adding one there must reach
    here. Bitwise, because the model runs in ``eval()`` with the generator re-seeded before each
    forward, so the single ``randn_like`` draw is the only RNG consumer and both runs share their
    $\\epsilon$."""
    model = _model(tiny_gated).eval()
    y_st, y_ph, u_stream = inputs
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise)

    assert len(_TARGET_ONLY_KEYS) == 7
    for key in _TARGET_ONLY_KEYS:
        assert torch.equal(base[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(base["source_state"], resampled["source_state"])


def test_no_gradient_path_runs_from_the_source_stream_to_the_prior(tiny_gated, inputs):
    """The stronger claim. Bitwise equality on one input is consistent with a pathway that exists
    and contributed zero; an autograd probe asks whether the pathway is in the graph at all."""
    model = _model(tiny_gated)
    y_st, y_ph, u_stream = inputs
    u_stream = u_stream.clone().requires_grad_(True)

    out = model(y_st, y_ph, u_stream)
    for key in ("mu_prior", "logvar_prior", "target_state", "mu_base"):
        (grad,) = torch.autograd.grad(
            out[key].sum(), u_stream, retain_graph=True, allow_unused=True
        )
        assert grad is None, f"{key} is differentiable with respect to the source stream"

    # The positive direction, so the ``grad is None`` assertions are not vacuous.
    (source_grad,) = torch.autograd.grad(
        out["source_state"].sum(), u_stream, retain_graph=True, allow_unused=True
    )
    assert source_grad is not None and float(source_grad.abs().max()) > 0.0


def test_the_source_pathway_receives_the_source_object_itself(tiny_kwargs, inputs):
    """Instrumented at the adapters -- the trust boundary where the raw streams enter -- through the
    sibling's own capture helper. Identity, not equality: an equal tensor could be the output of a
    concatenation that mixed a target block in and then happened to agree on this input.

    Asserted **ungated**, which is the only configuration in which identity is the right assertion:
    with a gate the forward hands the adapter the gate's output rather than the caller's object. The
    gated case is the next test.
    """
    model = _model(tiny_kwargs).eval()
    seen = _capture_adapter_inputs(model, inputs)

    assert model.source_gate is None and model.target_gate is None
    assert len(seen["source"]) == 1 and seen["source"][0] is inputs[2]
    assert len(seen["target"]) == 1
    assert torch.equal(seen["target"][0], torch.cat([inputs[0], inputs[1]], dim=-1))
    assert seen["target"][0].shape[-1] == model.c_y


def test_under_a_gate_each_adapter_receives_its_own_streams_survivors(tiny_gated, inputs):
    """The gated form of the same claim, which is what production runs.

    Identity no longer applies -- the gate gathers and delays first -- so what is asserted is that
    each adapter sees a tensor at *its own* stream's surviving width, and that the two widths differ.
    A concatenation that mixed the streams could not satisfy both: the source adapter's input would
    be wider than the source gate emits.
    """
    model = _model(tiny_gated).eval()
    seen = _capture_adapter_inputs(model, inputs)

    assert seen["source"][0].shape[-1] == model.source_gate.out_channels == 2
    assert seen["target"][0].shape[-1] == model.target_gate.out_channels == 3
    assert torch.equal(seen["source"][0], model.source_gate(inputs[2]))
    assert torch.equal(seen["target"][0], model.target_gate(torch.cat(inputs[:2], dim=-1)))


def test_the_two_encoders_share_no_parameter_tensor(tiny_gated):
    """Separate instances, not one module used twice: a shared encoder would make the source state
    a function of the target and every purity assertion above would be about the same tensor."""
    model = _model(tiny_gated)
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}
    source_ids = {id(parameter) for parameter in model.source_encoder.parameters()}

    assert target_ids and source_ids
    assert target_ids.isdisjoint(source_ids)


def test_the_forward_takes_no_target(tiny_kwargs, inputs):
    """The invariant this target domain adds, and the raw variant cannot need.

    The reconstruction target is now built from the *same kind of tensor the model is shown*, so the
    two could be confused in a way the raw variant's cannot -- and a forward that had somehow been
    handed the future would still return every contract shape. The forward takes three streams and
    holds no target: ``compute_loss`` gathers one, after the forward has run, from a stream the
    caller passes it separately.
    """
    import inspect

    assert list(inspect.signature(SeqVaeLagAttnTrfFs.forward).parameters) == [
        "self", "y_st", "y_ph", "u_stream",
    ]

    model = _model(tiny_kwargs).eval()
    with torch.no_grad():
        out = model(*inputs)

    # The one key whose name contains "target" is the encoder's history state, at d_model rather
    # than at the decoder's width -- so no returned tensor could be a forecast target.
    assert [key for key in out if "target" in key] == ["target_state"]
    assert out["target_state"].shape[-1] == model.d_model != model.decoder_out_channels


@pytest.mark.parametrize(
    "probe", ["bitwise", "autograd", "adapter-identity"],
)
def test_the_cross_wired_control_fails_every_purity_probe(tiny_gated, inputs, probe):
    """A model that genuinely mixes the streams must fail all three, or none of them is testing
    anything. The control is the sibling's, composed with this target domain."""
    model = _model(tiny_gated, cls=_CrossWiredFeatureModel).eval()
    y_st, y_ph, u_stream = inputs

    if probe == "bitwise":
        noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
        torch.manual_seed(0)
        with torch.no_grad():
            base = model(y_st, y_ph, u_stream)
        torch.manual_seed(0)
        with torch.no_grad():
            resampled = model(y_st, y_ph, noise)
        assert not torch.equal(base["mu_prior"], resampled["mu_prior"])
    elif probe == "autograd":
        leaky = u_stream.clone().requires_grad_(True)
        out = model(y_st, y_ph, leaky)
        (grad,) = torch.autograd.grad(out["mu_prior"].sum(), leaky, allow_unused=True)
        assert grad is not None
    else:
        assert _capture_adapter_inputs(model, inputs)["source"][0] is not inputs[2]


def test_the_control_is_this_target_domain_and_not_the_raw_one(tiny_gated):
    """A control built on the wrong parent would exercise a $16$-wide decoder, so the composition is
    checked rather than assumed."""
    model = _model(tiny_gated, cls=_CrossWiredFeatureModel)

    assert model.decoder_out_channels == 3
    assert _CrossWiredFeatureModel.forward is CrossWiredModel.forward
    assert _CrossWiredFeatureModel.compute_loss is SeqVaeLagAttnTrfFs.compute_loss


# ---------------------------------------------------------------------------------------
# 2. No decoder bypass
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("forecast", ["mu_base", "mu_full"])
@pytest.mark.parametrize("encoder", ["target_encoder", "source_encoder"])
def test_with_z_detached_no_gradient_reaches_either_encoder(tiny_gated, inputs, forecast, encoder):
    """The detach happens inside the model's real forward -- the sibling's helper wraps the sampling
    method -- so the probe covers the wiring as built, including both decoder invocations, rather
    than a hand-assembled call into the decoder."""
    model = _model(tiny_gated)
    _detach_latents(model)

    out = model(*inputs)
    parameters, grads = _encoder_grads(model, out[forecast], encoder)
    named = list(getattr(model, encoder).named_parameters())
    leaked = [name for (name, _), grad in zip(named, grads) if grad is not None]

    assert parameters, f"{encoder} has no parameters; the probe is vacuous"
    assert not leaked, f"{encoder} parameters reachable from {forecast} around z: {leaked}"


@pytest.mark.parametrize(
    "forecast, encoder",
    [("mu_base", "target_encoder"), ("mu_full", "source_encoder")],
    ids=["base-through-target", "full-through-source"],
)
def test_without_the_detach_the_same_probe_finds_gradients(tiny_gated, inputs, forecast, encoder):
    """Both halves of the positive direction. The base forecast cannot show the source half -- the
    source is only ever read by the posterior -- so ``mu_full`` is where its gradient has to
    appear."""
    model = _model(tiny_gated)
    out = model(*inputs)

    _, grads = _encoder_grads(model, out[forecast], encoder)
    named = list(getattr(model, encoder).named_parameters())
    unreached = [name for (name, _), grad in zip(named, grads) if grad is None]

    assert not unreached, f"{encoder} unreachable from {forecast} even through z: {unreached}"


def test_the_decoder_reads_a_latent_and_nothing_wider(tiny_gated, inputs):
    """The surface claim beside the autograd one. The decoder's first linear reads $d_z$, so no
    $d_{\\mathrm{model}}$-wide encoder state could be fed to it even if one were passed -- and the
    contrast is sharper here than in the raw variant, where the *output* width happens to equal a
    geometry constant."""
    model = _model(tiny_gated).eval()
    with torch.no_grad():
        out = model(*inputs)

    assert "decoder_state" not in out
    assert "delta_mu_src" not in out
    assert model.decoder.proj.body[0].in_features == model.d_z
    assert model.decoder.out_channels == model.decoder_out_channels != model.d_z


# ---------------------------------------------------------------------------------------
# 3. One shared decoder, invoked twice
# ---------------------------------------------------------------------------------------
def test_the_decoder_is_one_module_invoked_twice(tiny_gated, inputs):
    """Counted at the module, not argued from the source: two calls per forward, each taking one
    tensor, into the same object. Two decoders -- or one decoder given a second argument -- would
    make the base-minus-full gap a comparison of two functions rather than of two latents."""
    model = _model(tiny_gated).eval()
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
    assert not any(hasattr(model, name) for name in ("residual_decoder", "baseline_decoder"))


def test_the_decoder_subtree_holds_no_dropout_module_at_all(tiny_gated):
    """What makes the two invocations comparable in train mode: two dropout masks would put noise
    into every base-minus-full readout, and the gap would be reported as coupling.

    Stated as the absence of the *module*, which is what is actually true and is stronger than
    $p = 0$: the projection MLP builds its ``nn.Dropout`` layers only at a positive rate, so at the
    decoder's hard-coded $0$ there is nothing there to be re-enabled by a later ``model.train()`` or
    by a rate written into a config. Measured at a model built at $0.1$ throughout, so the absence
    is a choice rather than a consequence of a dropout-free model.
    """
    model = _model(tiny_gated, dropout=0.1)

    in_decoder = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and name.startswith(("decoder.", "horizon_core."))
    ]
    assert in_decoder == [], in_decoder
    assert model.lag_attn.attn_dropout.p == 0.0
    # And the encoders *did* receive the configured value, so the absence above was chosen.
    assert {
        module.p
        for module in model.target_encoder.modules()
        if isinstance(module, torch.nn.Dropout)
    } == {0.1}
    assert any(isinstance(module, torch.nn.Dropout) for module in model.modules())


# ---------------------------------------------------------------------------------------
# 4. Exact zero KL at initialisation
# ---------------------------------------------------------------------------------------
def test_the_kl_is_exactly_zero_at_init(tiny_kwargs, inputs):
    """Under the conv-Transformer suite's ``TINY_KWARGS``: ``base_decode='sample'`` and
    ``posterior_logvar_mode='residual'``, the constructor defaults, neither of which that keyword
    set names. The two shipped flags are covered separately below."""
    assert "base_decode" not in tiny_kwargs
    assert "posterior_logvar_mode" not in tiny_kwargs
    out = _train_mode_forward(tiny_kwargs, inputs)

    assert float(_closed_form_kl(out).abs().max()) == 0.0
    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert float(out["source_kl_lag_map"].abs().max()) == 0.0
    assert float(out["kld_per_t_per_head"].abs().max()) == 0.0


def test_the_two_forecasts_are_bitwise_identical_at_init_in_train_mode(tiny_gated, inputs):
    """One module, two invocations, two dropout masks -- unless the decoder has no dropout, which is
    what makes this exact. Train mode is the point: it is what turns the base-minus-full readout
    into a noise-free null. Asserted on the *gated* set, so the identity holds at the width the
    target gate resolves rather than only at the declared one."""
    out = _train_mode_forward(tiny_gated, inputs)

    assert torch.equal(out["mu_post"], out["mu_prior"])
    assert torch.equal(out["logvar_post"], out["logvar_prior"])
    assert torch.equal(out["z_prior"], out["z_post"])
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])
    assert out["mu_base"].shape[-1] == 3


def test_the_encoder_dropout_is_actually_active_in_train_mode(tiny_gated, inputs):
    """The control for the paragraph in this file's docstring: if dropout were inert in train mode,
    every identity above would hold for reasons that say nothing about one common encoder forward."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**dict(tiny_gated, dropout=0.1)).train()

    torch.manual_seed(1)
    first = model(*inputs)["target_state"]
    torch.manual_seed(2)
    second = model(*inputs)["target_state"]

    assert not torch.equal(first, second), "dropout is not active; the train-mode claims are empty"


def test_everything_above_becomes_false_once_perturbed(tiny_gated, inputs, perturb_posterior):
    """The zero must be a property of the init, not of the model being unable to produce a KL.
    Without this, a model whose KL was structurally stuck at zero -- a broken posterior, a detached
    graph, a source encoder returning a constant -- would pass every test above."""
    out = _train_mode_forward(tiny_gated, inputs, perturb=perturb_posterior)

    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert float(out["kld_per_t"].abs().max()) > _TOL
    assert not torch.equal(out["z_prior"], out["z_post"])
    assert not torch.equal(out["mu_base"], out["mu_full"])


def test_under_the_shipped_posterior_mode_the_init_kl_is_still_exactly_zero(tiny_gated, inputs):
    """``posterior_logvar_mode: independent`` writes the posterior log-variance from its own head
    rather than as a residual, so the zero is bought by ``head_init_calibration: true`` -- which puts
    that head and the prior's on the same value -- rather than by the residual being zero. The
    shipped configuration sets both, and the pairing is what makes the KL start at zero."""
    out = _train_mode_forward(
        dict(tiny_gated, posterior_logvar_mode="independent", head_init_calibration=True), inputs
    )

    assert float(_closed_form_kl(out).abs().max()) == 0.0
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


def test_under_the_shipped_base_decode_the_forecasts_are_not_bitwise_identical(tiny_gated, inputs):
    """``base_decode: mean`` decodes the base branch from $\\mu^p$ rather than from a draw, so at
    init the two forecasts differ by exactly the sampling noise $z^q$ carries. Stated here because a
    reader who took the bitwise identity above as unconditional would read the shipped run's
    non-zero init ``pred_gap`` as a bug."""
    out = _train_mode_forward(dict(tiny_gated, base_decode="mean"), inputs)

    assert float(_closed_form_kl(out).abs().max()) == 0.0  # the KL is still exactly zero
    assert torch.equal(out["z_prior"], out["mu_prior"])
    assert not torch.equal(out["mu_base"], out["mu_full"])


def test_the_invariants_hold_at_the_production_geometry_and_budget():
    """One pass at the real thing. The tiny fixture's guard is hand-made; the shipped one gathers
    $78$ of $109$ target channels and delays every survivor, and source purity has to survive
    that."""
    kwargs = shipped_gated_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**dict(kwargs, dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(0)
    length = int(kwargs["sequence_length"])
    y_st = torch.randn(BATCH, length, 43, generator=generator)
    y_ph = torch.randn(BATCH, length, 66, generator=generator)
    u_stream = torch.randn(BATCH, length, 58, generator=generator)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, torch.randn(u_stream.shape, generator=generator))

    for key in _TARGET_ONLY_KEYS:
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) == 0.0
    assert reference["mu_base"].shape[-1] == 78


# ---------------------------------------------------------------------------------------
# The initialisation policies, at a 78-wide output head
# ---------------------------------------------------------------------------------------
def test_the_logvar_head_is_calibrated_across_all_seventy_eight_channels():
    r"""The one initialisation policy the width change actually reaches.

    $\log(5/3)$ is the exact pre-image of log-variance $0$ under ``smooth_bound(-5, 3)``: $\sigma = 1$
    in z-scored units, the trivial predictor's variance. It must hold on **every** one of the $78$
    emitted channels, not on the $16$ a raw head has: an uncalibrated tail would put those channels'
    initial NLL orders of magnitude above the trivial predictor's, and at $78$ channels there is far
    more tail to get wrong.

    Exactness is the point. ``smooth_bound`` is a sigmoid, so a merely shrunk head would start near
    zero only on average rather than per coordinate, and the summed block would still be wrong.
    """
    model = _model(shipped_gated_kwargs(), head_init_calibration=True)
    bias = model.decoder.logvar_head.bias
    lo, hi = model.logvar_clamp

    assert bias.numel() == model.decoder_out_channels == 78
    assert torch.allclose(bias, torch.full_like(bias, math.log(5.0 / 3.0)))
    assert torch.allclose(smooth_bound(bias, lo, hi), torch.zeros_like(bias), atol=1e-6)


def test_the_uncalibrated_head_is_not_already_at_the_trivial_predictor():
    """The negative control: without the policy the log-variance bias is the generic pass's zero,
    which ``smooth_bound`` maps far from $0$, so the calibrated assertion is not vacuous."""
    model = _model(shipped_gated_kwargs(), head_init_calibration=False)
    lo, hi = model.logvar_clamp

    assert float(smooth_bound(model.decoder.logvar_head.bias, lo, hi).abs().min()) > 0.5


def test_the_calibration_reaches_the_wide_head_because_it_runs_after_construction(tiny_gated):
    """What dates the decoder against the init block, restated where the width changed.

    The decoder is built from the width hook *before* the generic initialisation, the depthwise
    repair and the two calibration passes. If it were built after any of them, the calibration would
    have run on a head of a different width -- and the assertion above would be about a tensor the
    policy never touched. Asserted as the two widths agreeing at a gate size the raw variant would
    never produce.
    """
    model = _model(tiny_gated, head_init_calibration=True)

    assert model.decoder_out_channels == 3 != model.raw_per_step
    assert model.decoder.logvar_head.bias.numel() == 3
    assert torch.allclose(
        model.decoder.logvar_head.bias,
        torch.full_like(model.decoder.logvar_head.bias, math.log(5.0 / 3.0)),
    )


def test_the_depthwise_repair_still_ran_at_the_new_width():
    """The policy the width must *not* have moved. ``n_depthwise_init`` counts one stem convolution
    per kernel per stream, and it is the only evidence the variance-preserving pass was not a silent
    no-op -- which is what an init-order change would make it."""
    model = _model(shipped_gated_kwargs())

    assert model.n_depthwise_init == 2 * len(model.target_encoder.conv_blocks) == 4


def test_the_film_generators_and_delta_heads_are_still_exactly_zero(tiny_gated):
    """The other two zeroings the generic pass would otherwise have undone. Both are what make the
    exact zero-KL start and the identity-at-init decoder true, and both run after the decoder is
    built -- so a width change that had moved the construction point would show up here."""
    model = _model(tiny_gated)

    film = model.horizon_core.refine.film
    assert film is not None and len(film) > 0
    for generator in film:
        assert float(generator.weight.abs().max()) == 0.0
    for name in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(model.posterior_head, name)
        layers = list(module) if isinstance(module, torch.nn.ModuleList) else [module]
        for layer in layers:
            assert float(layer.weight.abs().max()) == 0.0, f"{name} weight not zeroed"


def test_the_full_shipped_flag_set_starts_at_exactly_zero_kl(tiny_gated, inputs):
    """The three policies together, at the tiny geometry so the forward is cheap. A flag bundle that
    broke the zero-KL start would make every ``pred_gap`` reading of the first epochs meaningless."""
    out = _train_mode_forward(
        dict(tiny_gated, horizon_embed_std=0.8, head_init_calibration=True, a_head_gain=2.0),
        inputs,
    )

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])
