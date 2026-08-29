r"""The structural invariants, without which the readout means nothing.

Four properties make $\mathrm{KL}(q_t \Vert p_t)$ readable as "what the source added":

1. **Source purity.** The prior, and everything decoded from $z^p$, is a function of the target's
   history alone. Replace the source with noise and neither may move.
2. **No decoder bypass.** Gradient reaches the decoder only through $z$. Without it the latent could
   shrink into a residual code on a target-derived shortcut, and reading $\mu^p$ as the predictive
   state would be unsupported.
3. **One shared decoder, invoked twice.** Base and full differ only in which latent they were given
   -- same module, same weights, same (absent) dropout.
4. **Exact zero KL at initialisation**, which lives in ``test_zero_kl_init.py``.

All four are inherited. What is *not* inherited is the decode itself: this model gathers the latents
at a tiled anchor set before invoking the decoder, and a gather is where a batch axis and an anchor
axis could be transposed, or where the two branches could be handed different rows. So each test
below runs against **both** classes wherever the fixture allows -- the causal model and the two-sided
feature sibling it claims to inherit from -- which turns "the subclass inherits the invariant" from
an argument about class hierarchies into a measurement.

The forward signature is pinned as a literal parameter list, as in every sibling's copy. This one
carries two more names, and their presence in the *signature* is the design decision: the anchor
geometry is an argument rather than something read off ``self.training``, because the diagnostic
callback calls ``eval()`` during training and then the objective -- so a mode-derived geometry would
make ``total_loss`` a function of the dropout switch.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    TINY_STRIDE,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs

#: Both classes, so every invariant is asserted on this model *and* on the one it inherits it from.
#: The sibling takes no anchor arguments, so the call sites splat a per-class argument list.
_CLASSES = (SeqVaeLagAttnFs, SeqVaeLagAttnCfs)
_CLASS_IDS = ("fs", "cfs")

_TOL = 1e-6


def _kwargs_for(cls) -> dict:
    """The tiny guarded keyword set, with this family's keywords removed for the sibling."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    if cls is SeqVaeLagAttnCfs:
        return kwargs
    return {
        name: value
        for name, value in kwargs.items()
        if name not in ("target_warmup_steps", "source_warmup_steps", "anchor_stride")
    }


def _extra_args(cls) -> tuple:
    """The anchor arguments, which only this family's forward takes."""
    return (1, TINY_STRIDE) if cls is SeqVaeLagAttnCfs else ()


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
    # Post-gate on both sides here, unlike the ungated sibling suites: this family always gathers.
    assert seen["source"][0].shape[-1] == model.source_gate.out_channels
    assert seen["target"][0].shape[-1] == model.target_gate.out_channels


def test_the_forward_takes_no_target() -> None:
    """The invariant this target domain adds.

    The reconstruction target is now the same *kind* of tensor the model is shown, so the two could
    be confused in a way the raw sibling's cannot -- and a forward that had been handed the future
    would still return every contract shape. The forward takes three streams and two integers; it is
    ``compute_loss`` that gathers a target afterwards, from a stream the caller passes separately.
    """
    parameters = list(inspect.signature(SeqVaeLagAttnCfs.forward).parameters)
    assert parameters == ["self", "y_st", "y_ph", "u_stream", "anchor_phase", "anchor_stride"]

    model = _model(SeqVaeLagAttnCfs).eval()
    with torch.no_grad():
        out = model(*make_streams(_kwargs_for(SeqVaeLagAttnCfs)), 1, TINY_STRIDE)
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

    And on this model, the two calls must receive the **same** anchor rows -- two gathers at two
    indices would make the base-minus-full gap a comparison of two anchor sets.
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
    """One pass at the real thing: $300$ steps, $98$ of $102$ target channels, eleven tiles.

    The tiny fixture's guard is hand-built; this one is resolved from the committed shard, so the
    invariants are asserted against the geometry a run would actually train at.
    """
    kwargs = shipped_warmup_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnCfs(**dict(kwargs, dropout=0.0)).eval()
    y_st, y_ph, u_stream = make_streams(kwargs, batch=BATCH)
    phase = torch.tensor([0, 7])

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, phase)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(
            y_st, y_ph, torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(7)),
            phase,
        )

    for key in ("mu_prior", "logvar_prior", "z_prior", "mu_base", "logvar_base"):
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) == 0.0
    assert tuple(reference["mu_base"].shape) == (BATCH, 5, 30, 98)


# =================================================================================================
# 1b. Source purity, restated: the prior sees no function of the source's VALUES
#
# The invariant above is asserted on a model whose prior conditions on the target history alone. The
# shipped model's prior additionally conditions on a CLOCK, and the invariant is therefore restated
# rather than weakened: what the prior may not see is a function of the source's *values*, and the
# clock is a function of $t$ and the configuration alone.
#
# The clock is the encode of a stream that is exactly zero -- identical for every recording, under
# every intervention on the source, and by construction carrying nothing the source said. What it
# does depend on is the source pathway's own parameters, which is why it is detached: gradient must
# not couple the two pathways either.
#
# So the restatement is checked in four parts, each of which a plausible implementation could fail
# on its own: the prior does not move when the source's values do; the clock is not identically the
# same row at every step (which is what made the availability staircase inert); it is the same
# tensor in train mode and in eval mode; and no gradient reaches the source pathway through it.
# =================================================================================================
def _clocked(**overrides):
    """This cell's model with the prior's clock on, at the tiny guarded geometry."""
    return _model(SeqVaeLagAttnCfs, prior_availability_input=True, **overrides)


def test_the_prior_does_not_move_when_the_sources_values_do_even_with_the_clock_on() -> None:
    """The restated invariant, measured. This is the same comparison as the purity test above and
    it is not redundant with it: there the prior takes one input, here it takes two, and the second
    is built from the source pathway. If the clock were an encode of the *actual* source rather
    than of silence, every tensor below would move and every shape would be unchanged."""
    model = _clocked().eval()
    y_st, y_ph, u_stream = make_streams(_kwargs_for(SeqVaeLagAttnCfs))
    extra = _extra_args(SeqVaeLagAttnCfs)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream, *extra)
    noise = torch.randn(u_stream.shape, generator=torch.Generator().manual_seed(99))
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, noise, *extra)

    for key in (
        "mu_prior", "logvar_prior", "raw_logvar_prior", "target_state", "z_prior",
        "mu_base", "logvar_base",
    ):
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])


def test_the_prior_clock_is_the_encode_of_silence_and_of_nothing_else() -> None:
    """What the clock *is*, asserted against the tensor the source-null control feeds the posterior.

    The two have to be the same object for the cancellation argument to hold at all: the prior and
    the null posterior are supposed to receive the same input, so their divergence is learnable to
    zero rather than floored by an asymmetry. Built here by hand -- gate, then the configured
    key/value pathway, over an exactly zero stream -- rather than read off the model, so a clock
    that started encoding something else would fail rather than agree with itself.
    """
    model = _clocked().eval()
    _, _, u_stream = make_streams(_kwargs_for(SeqVaeLagAttnCfs))

    clock = model._prior_clock(u_stream)
    zeros = u_stream.new_zeros((1, *u_stream.shape[1:]))
    with torch.no_grad():
        expected = model.encode_source_kv(
            zeros if model.source_gate is None else model.source_gate(zeros)
        )

    assert torch.equal(clock, expected)
    assert clock.shape == (1, model.sequence_length, model.d_model)


def test_the_prior_clock_is_not_the_same_row_at_every_scored_step() -> None:
    r"""The failure the availability staircase had, and the reason the clock is an encode.

    $\mathbb 1[t \ge W'_c + d_c]$ is provably constant over every *scored* anchor -- the constructor
    refuses any floor below $\max_c(W'_c + d_c)$, which is the last step at which it changes -- so a
    prior conditioned on it gains an offset its biases already span. A clock that had quietly become
    constant again would still be the right shape, still be detached, still pass every test above,
    and condition the prior on nothing. So the live row count over the scored range is asserted
    directly.
    """
    model = _clocked().eval()
    _, _, u_stream = make_streams(_kwargs_for(SeqVaeLagAttnCfs))

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
    model = _clocked(dropout=0.3, source_dropout=0.3)
    _, _, u_stream = make_streams(_kwargs_for(SeqVaeLagAttnCfs))

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
    detaching it is what keeps the two pathways' gradients uncoupled -- the same reason the prior's
    projection is not shared with the adapter's ``mask_proj``.

    Probed through the prior alone. The source modules still receive gradient from the matched
    forward in the same step, so nothing leaves the distributed run's expectation set; what must not
    exist is a path from the *prior's* output back into them.
    """
    model = _clocked().eval()
    _, _, u_stream = make_streams(_kwargs_for(SeqVaeLagAttnCfs))

    clock = model._prior_clock(u_stream)

    assert not clock.requires_grad
    assert clock.grad_fn is None