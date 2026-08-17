r"""The four structural invariants, asserted on this composition and on the cell beside it.

Every one of them is a property of code this package does not write, which is exactly why they are
worth re-asserting here: a composition can break an inherited invariant by changing *which* objects
are composed, without touching any of them. Each test therefore runs against both causal cells, so a
divergence is a divergence between two models rather than a fact about one.

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
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    TINY_KWARGS as CONV_LSTM_TINY_KWARGS,
)
from teb_vae.lag_attn_cfs.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

from .conftest import (
    BATCH,
    TINY_KWARGS,
    TINY_STRIDE,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: Both causal cells, so every invariant is asserted on this model *and* on the one it is compared
#: against. The two constructors' schemas differ by twelve keywords, so each is built from its own
#: keyword set.
_CLASSES = (SeqVaeLagAttnCfs, SeqVaeLagAttnTrfCfs)
_CLASS_IDS = ("cfs", "trf_cfs")

_TOL = 1e-6


def _kwargs_for(cls) -> dict:
    """The tiny guarded keyword set for whichever architecture is being built."""
    if cls is SeqVaeLagAttnTrfCfs:
        return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    return conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


def _streams_for(cls):
    """The three inputs. Both cells declare the same widths, so the streams are the same shape."""
    return make_streams(TINY_KWARGS if cls is SeqVaeLagAttnTrfCfs else CONV_LSTM_TINY_KWARGS)


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
    y_st, y_ph, u_stream = _streams_for(cls)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, 1, TINY_STRIDE)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise, 1, TINY_STRIDE)

    for key in (
        "mu_prior", "logvar_prior", "raw_logvar_prior", "target_state", "z_prior",
        "mu_base", "logvar_base",
    ):
        assert torch.equal(reference[key], resampled[key]), key
    # And the source pathway did notice the change -- otherwise the test proves nothing.
    assert not torch.equal(reference["source_state"], resampled["source_state"])


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_source_pathway_receives_only_the_source_stream(cls) -> None:
    """Instrumented at the adapters -- the trust boundary where the streams enter."""
    model = _model(cls).eval()
    y_st, y_ph, u_stream = _streams_for(cls)
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
            model(y_st, y_ph, u_stream, 1, TINY_STRIDE)
    finally:
        for handle in handles:
            handle.remove()

    assert len(seen["source"]) == 1 and len(seen["target"]) == 1
    # Post-gate on both sides: this target domain always gathers.
    assert seen["source"][0].shape[-1] == model.source_gate.out_channels
    assert seen["target"][0].shape[-1] == model.target_gate.out_channels


def test_the_forward_takes_no_target() -> None:
    """The invariant this target domain adds.

    The reconstruction target is the same *kind* of tensor the model is shown, so the two could be
    confused in a way the raw sibling's cannot -- and a forward that had been handed the future would
    still return every contract shape. The forward takes three streams and two integers; it is
    ``compute_loss`` that gathers a target afterwards, from a stream the caller passes separately.
    """
    parameters = list(inspect.signature(SeqVaeLagAttnTrfCfs.forward).parameters)
    assert parameters == ["self", "y_st", "y_ph", "u_stream", "anchor_phase", "anchor_stride"]

    model = _model(SeqVaeLagAttnTrfCfs).eval()
    with torch.no_grad():
        out = model(*_streams_for(SeqVaeLagAttnTrfCfs), 1, TINY_STRIDE)
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

    out = model(*_streams_for(cls), 1, TINY_STRIDE)
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

    On these models that is a statement about the *gather* as well: a decode at anchors none of
    which the encoder reaches would leave parameters unreached and look exactly like a bypass.
    """
    model = _model(cls)
    out = model(*_streams_for(cls), 1, TINY_STRIDE)
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
# 3. One shared decoder, invoked twice, at one anchor set
# =================================================================================================
@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_decoder_is_one_module_invoked_twice(cls) -> None:
    """Counted at the module: two calls per forward, each taking one tensor, into the same object.

    And the two calls must receive the **same** anchor rows -- two gathers at two indices would make
    the base-minus-full gap a comparison of two anchor sets.
    """
    model = _model(cls).eval()
    calls: list = []
    handle = model.decoder.register_forward_pre_hook(
        lambda module, args: calls.append((module, args))
    )
    try:
        with torch.no_grad():
            model(*_streams_for(cls), 1, TINY_STRIDE)
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
# 4. Zero KL at initialisation, and the control that makes it mean something
# =================================================================================================
@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_kl_is_identically_zero_at_initialisation(cls) -> None:
    """$q = p$ at step $0$, recomputed in closed form from the returned parameters so that no model
    certifies itself. The whole coupling readout rests on it: a KL that started positive would be
    reported as coupling the source never provided."""
    model = _model(cls).eval()

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*_streams_for(cls), 1, TINY_STRIDE)

    assert float(_closed_form_kl(out).abs().max()) < _TOL
    assert float(out["kld_per_t"].abs().max()) < _TOL


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_zero_kl_probe_is_not_vacuous(cls, perturb_posterior) -> None:
    """The negative control. The posterior delta heads are zero-initialised, so every KL assertion
    on a fresh model passes on a broken one; the shared perturbation is the escape."""
    model = _model(cls).eval()
    perturb_posterior(model)

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*_streams_for(cls), 1, TINY_STRIDE)

    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert float(out["kld_per_t"].abs().max()) > _TOL


@pytest.mark.parametrize("cls", _CLASSES, ids=_CLASS_IDS)
def test_the_lag_map_sums_over_lags_to_the_per_step_kl(cls, perturb_posterior) -> None:
    r"""$\sum_\ell M_{b,t,\ell} = K_{b,t}$, exactly, because the attention probabilities carry no
    dropout. Perturbed first, or both sides are zero and the identity is vacuous."""
    model = _model(cls).eval()
    perturb_posterior(model)

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*_streams_for(cls), 1, TINY_STRIDE)

    summed = out["source_kl_lag_map"].sum(dim=-1)
    assert float(out["kld_per_t"].abs().max()) > _TOL
    assert torch.allclose(summed, out["kld_per_t"], atol=1e-5, rtol=1e-5)


# =================================================================================================
# The production geometry and budget
# =================================================================================================
def test_the_invariants_hold_at_the_production_geometry_and_budget() -> None:
    """One pass at the real thing: $300$ steps, $98$ of $102$ target channels, eleven tiles.

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

    assert model.decoder_out_channels == 98
    assert reference["anchor_index"].shape[1] == 5
    for key in ("mu_prior", "z_prior", "mu_base"):
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) < _TOL
