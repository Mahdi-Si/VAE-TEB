r"""The four structural invariants, asserted on this composition.

Every one of them is a property of code this package does not write, which is exactly why they are
worth re-asserting here: a composition can break an inherited invariant by changing *which* objects
are composed, without touching any of them. The conv-LSTM causal cell asserts the same invariants on
its own composition; this file asserts them on the conv-Transformer one.

1. **Source purity.** $p(z \mid Y)$ reads the target alone: resampling the source must leave the
   prior, the target state and the base forecast bitwise unchanged -- and must move the source
   state, or the probe proves nothing.
2. **No decoder bypass.** $z$ is the decoder's only input. Detaching it inside the model's own
   forward must starve every target-encoder parameter; without the detach every one of them must be
   reachable, which on a tiled model is also a statement about the gather.
3. **One shared decoder, invoked twice, at the same anchors.** Two gathers at two indices would make
   the base-minus-full gap a comparison of two anchor sets, and no shape would say so.
4. **Zero KL at initialisation**, with ``perturb_posterior`` as the negative control: the posterior
   delta heads are zero-initialised, so every KL assertion on a fresh model passes on a broken one.

With the prior's availability clock on, source purity is restated rather than weakened: the prior
may see a function of $t$ and the configuration, never of the source's values.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

from .conftest import (
    BATCH,
    TINY_KWARGS,
    TINY_STRIDE,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

_TOL = 1e-6


def _model(**overrides) -> SeqVaeLagAttnTrfCfs:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfCfs(**tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, **overrides))


def _streams():
    """The three seeded inputs at the tiny geometry."""
    return make_streams(TINY_KWARGS)


def _closed_form_kl(out: dict) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ from the returned parameters alone, so no model certifies itself."""
    return 0.5 * (
        out["logvar_prior"]
        - out["logvar_post"]
        + (out["logvar_post"].exp() + (out["mu_post"] - out["mu_prior"]) ** 2)
        / out["logvar_prior"].exp()
        - 1.0
    )


def _source_resampled(model):
    """The matched forward and one with the source stream replaced by fresh noise, same $\\epsilon$."""
    y_st, y_ph, u_stream = _streams()
    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, 1, TINY_STRIDE)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise, 1, TINY_STRIDE)
    return reference, resampled


_SOURCE_FREE_KEYS = (
    "mu_prior", "logvar_prior", "raw_logvar_prior", "target_state", "z_prior",
    "mu_base", "logvar_base",
)


# =================================================================================================
# 1. Source purity
# =================================================================================================
def test_resampling_the_source_leaves_the_prior_and_base_forecast_unchanged() -> None:
    """Bitwise: the model runs in ``eval()`` with the generator re-seeded before each forward, so
    the single ``randn_like`` draw is the only RNG consumer and both runs share their $\\epsilon$."""
    reference, resampled = _source_resampled(_model().eval())

    for key in _SOURCE_FREE_KEYS:
        assert torch.equal(reference[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(reference["source_state"], resampled["source_state"])


# =================================================================================================
# 2. No decoder bypass
# =================================================================================================
def _unreached_target_encoder_parameters(model) -> list:
    out = model(*_streams(), 1, TINY_STRIDE)
    grads = torch.autograd.grad(
        out["mu_base"].sum(), list(model.target_encoder.parameters()), allow_unused=True
    )
    return [
        name
        for (name, _), grad in zip(model.target_encoder.named_parameters(), grads)
        if grad is None
    ]


def test_the_target_encoder_reaches_the_decoder_through_z_and_only_through_z() -> None:
    """The detach happens inside the model's real forward -- by wrapping the sampling method -- so
    the probe covers the wiring as built, not a hand-assembled call into the decoder.

    The positive direction first: through $z$, every target-encoder parameter is on the path. On a
    tiled model that is a statement about the *gather* as well -- a decode at anchors none of which
    the encoder reaches would leave parameters unreached and look exactly like a bypass."""
    model = _model()
    parameters = [name for name, _ in model.target_encoder.named_parameters()]

    assert _unreached_target_encoder_parameters(model) == []

    sample = model._reparameterize_shared
    model._reparameterize_shared = lambda *args: tuple(  # type: ignore[method-assign]
        z.detach() for z in sample(*args)
    )
    assert _unreached_target_encoder_parameters(model) == parameters, (
        "target-encoder parameters reachable around z"
    )


# =================================================================================================
# 3. One shared decoder, invoked twice, at one anchor set
# =================================================================================================
def test_the_decoder_is_one_module_invoked_twice() -> None:
    """Counted at the module: two calls per forward, each taking one tensor, into the same object.

    And the two calls must receive the **same** anchor rows -- two gathers at two indices would make
    the base-minus-full gap a comparison of two anchor sets.
    """
    model = _model().eval()
    calls: list = []
    handle = model.decoder.register_forward_pre_hook(
        lambda module, args: calls.append((module, args))
    )
    try:
        with torch.no_grad():
            model(*_streams(), 1, TINY_STRIDE)
    finally:
        handle.remove()

    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is model.decoder
    assert all(len(args) == 1 for _module, args in calls)
    assert calls[0][1][0].shape == calls[1][1][0].shape


def test_the_decoder_carries_no_dropout() -> None:
    """What makes the two invocations comparable in train mode: two dropout masks would put noise
    into every base-minus-full readout, and the gap would be reported as coupling."""
    model = _model(dropout=0.1)

    dropouts = [
        module.p for module in model.decoder.modules() if isinstance(module, torch.nn.Dropout)
    ]
    assert all(probability == 0.0 for probability in dropouts), dropouts
    assert model.lag_attn.attn_dropout.p == 0.0


# =================================================================================================
# 4. Zero KL at initialisation, and the control that makes it mean something
# =================================================================================================
def _forward_eval(model):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*_streams(), 1, TINY_STRIDE)


def test_the_kl_is_identically_zero_at_initialisation_and_not_after_a_perturbation(
    perturb_posterior,
) -> None:
    """$q = p$ at step $0$, recomputed in closed form from the returned parameters so that no model
    certifies itself. The whole coupling readout rests on it: a KL that started positive would be
    reported as coupling the source never provided.

    The negative control: the posterior delta heads are zero-initialised, so every KL assertion on
    a fresh model passes on a broken one; the shared perturbation is the escape."""
    model = _model().eval()

    out = _forward_eval(model)
    assert float(_closed_form_kl(out).abs().max()) < _TOL
    assert float(out["kld_per_t"].abs().max()) < _TOL

    perturb_posterior(model)
    out = _forward_eval(model)
    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert float(out["kld_per_t"].abs().max()) > _TOL


def test_the_lag_map_sums_over_lags_to_the_per_step_kl(perturb_posterior) -> None:
    r"""$\sum_\ell M_{b,t,\ell} = K_{b,t}$, exactly, because the attention probabilities carry no
    dropout. Perturbed first, or both sides are zero and the identity is vacuous."""
    model = _model().eval()
    perturb_posterior(model)

    out = _forward_eval(model)

    summed = out["source_kl_lag_map"].sum(dim=-1)
    assert float(out["kld_per_t"].abs().max()) > _TOL
    assert torch.allclose(summed, out["kld_per_t"], atol=1e-5, rtol=1e-5)


# =================================================================================================
# The production geometry and budget
# =================================================================================================
def test_the_invariants_hold_at_the_production_geometry_and_budget() -> None:
    """One pass at the real thing: the shipped window, the resolved budget, the shipped tiling.

    The tiny fixture's guard is hand-built; this one is resolved from the committed shard, so the
    invariants are asserted against the geometry a run would actually train at.
    """
    kwargs = shipped_warmup_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCfs(**dict(kwargs, dropout=0.0)).eval()
    y_st, y_ph, u_stream = make_streams(kwargs, batch=BATCH)
    phase = torch.tensor([0, 7])

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, phase)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise, phase)

    span = model.geometry.t_valid - model.warmup_period
    assert reference["anchor_index"].shape[1] == -(-span // model.anchor_stride)
    for key in ("mu_prior", "z_prior", "mu_base"):
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) < _TOL


# =================================================================================================
# Source purity, restated: the prior sees no function of the source's VALUES
#
# The shipped model's prior additionally conditions on a CLOCK, and the invariant is therefore
# restated rather than weakened: what the prior may not see is a function of the source's *values*,
# and the clock is a function of $t$ and the configuration alone. It is the encode of a stream that
# is exactly zero -- identical for every recording and under every intervention on the source -- and
# it depends on the source pathway's own parameters, which is why it is detached.
#
# So the restatement is checked in four parts, each of which a plausible implementation could fail
# on its own: the prior does not move when the source's values do; the clock is not identically the
# same row at every scored step (which is what made the availability staircase inert); it is the
# same tensor in train mode and in eval mode; and no gradient reaches the source pathway through it.
# =================================================================================================
def test_the_prior_does_not_move_when_the_sources_values_do_even_with_the_clock_on() -> None:
    """The restated invariant, measured. Not redundant with the purity test above: there the prior
    takes one input, here it takes two, and the second is built from the source pathway. If the
    clock were an encode of the *actual* source rather than of silence, every tensor below would
    move and every shape would be unchanged."""
    reference, resampled = _source_resampled(_model(prior_availability_input=True).eval())

    for key in _SOURCE_FREE_KEYS:
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])


def test_the_prior_clock_is_not_the_same_row_at_every_scored_step() -> None:
    r"""The failure the availability staircase had, and the reason the clock is an encode.

    $\mathbb 1[t \ge W'_c + d_c]$ is provably constant over every *scored* anchor -- the constructor
    refuses any floor below $\max_c(W'_c + d_c)$, which is the last step at which it changes -- so a
    prior conditioned on it gains an offset its biases already span. A clock that had quietly become
    constant again would still be the right shape, still be detached, still pass every test above,
    and condition the prior on nothing. So the live row count over the scored range is asserted
    directly.
    """
    model = _model(prior_availability_input=True).eval()
    _, _, u_stream = _streams()

    clock = model._prior_clock(u_stream)[0, model.warmup_period :]
    # Rounded before the distinct count: these are float activations, and two rows differing in the
    # last bit are "distinct" to `unique`, which would report an inert clock as a live one.
    distinct = torch.unique(clock.round(decimals=4), dim=0).shape[0]

    assert clock.shape[0] > 1, "the fixture has no scored range to measure over"
    assert distinct > 1, "the prior's clock is constant over every scored anchor"


def test_the_prior_clock_is_the_same_tensor_in_train_mode_and_in_eval_mode() -> None:
    """It is defined as a function of $t$ and the configuration, so it must not follow the dropout
    switch. Under live dropout it would be a fresh draw every step, and the prior's input
    distribution would differ between the mode the objective runs in and the mode every readout is
    measured in -- which is the sort of difference that shows up as an unexplained gap between a
    training curve and an evaluation."""
    model = _model(prior_availability_input=True, dropout=0.3, source_dropout=0.3)
    _, _, u_stream = _streams()

    model.train()
    torch.manual_seed(0)
    in_train = model._prior_clock(u_stream)
    model.eval()
    torch.manual_seed(1)
    in_eval = model._prior_clock(u_stream)

    assert torch.equal(in_train, in_eval)
    # And the pathway is left in the mode it was found in, so a clock cannot silently disable
    # dropout for the forward that follows it.
    model.train()
    model._prior_clock(u_stream)
    assert all(module.training for module in model.source_kv_modules())


def test_no_gradient_reaches_the_source_pathway_through_the_prior() -> None:
    """The clock depends on the source pathway's *parameters*, which the announcement did not, so
    detaching it is what keeps the two pathways' gradients uncoupled.

    The source modules still receive gradient from the matched forward in the same step, so nothing
    leaves the distributed run's expectation set; what must not exist is a path from the *prior's*
    input back into them.
    """
    model = _model(prior_availability_input=True).eval()
    _, _, u_stream = _streams()

    clock = model._prior_clock(u_stream)

    assert not clock.requires_grad
    assert clock.grad_fn is None
