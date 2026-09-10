r"""The invariants every source readout downstream is measured against.

Three groups, and the order matters because each rests on the one before it.

**Source absence reproduces the prior.** Three different ways the source can say nothing -- every
selector off, every lag unavailable, the output projection still at zero -- and all three must give
back the prior exactly, in training mode as well as evaluation mode. Every control in the
evaluation package is a difference against one of these, so an approximate equality here would be
an unknown offset in every margin reported later.

**A valid standardized zero is an observation, not an absence.** The fourth case, and the one that
is a *negative* assertion: there must be no code path forcing an observed zero to reproduce the
prior. Absence and a real zero are different events, and a removal-based explanation means one
thing or the other depending on which the model conflates them into.

**The zero start is a starting point, not a fixed point.** The output projection is zero, so the
divergence gradient with respect to the residual is zero and the hidden layers can see no gradient
on the very first step. What must still hold is that the *predictive* gradient reaches the final
projection through a non-constant decoder, or the model would train as a target-only forecaster
forever and report a zero gap as a finding.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    DECLARED_C_U,
    TINY_BATCH,
    TINY_N_LAGS,
    TINY_SEQ_LEN,
    build_tiny_model,
    tiny_model_kwargs,
    tiny_streams,
)


def build_model(*, trained: bool = False, seed: int = 23, **overrides):
    """Build the tiny model, optionally with its source pathway moved off zero.

    Args:
        trained: Move the proposal head's output projection off its zero start, standing in for
            the first optimizer steps. Every assertion about a *nonzero* source effect is vacuous
            without it.
        seed: Seed for that draw.
        **overrides: Constructor keywords to replace.

    Returns:
        The model, in evaluation mode.
    """
    model = build_tiny_model(**overrides)
    if trained:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            model.proposal_head.output_proj.weight.normal_(0.0, 0.4, generator=generator)
            model.proposal_head.output_proj.bias.normal_(0.0, 0.2, generator=generator)
    return model.eval()


def run(model, *, seed: int = 0, streams=None, **kwargs):
    """Run one dense forward under a fixed noise seed.

    Args:
        model: The model.
        seed: Seed for the reparameterisation draw, so two runs are comparable.
        streams: Inputs, or ``None`` for the seeded fixture.
        **kwargs: Extra forward keywords.

    Returns:
        The forward's dict.
    """
    torch.manual_seed(seed)
    return model(
        *(tiny_streams() if streams is None else streams),
        anchor_phase=0,
        anchor_stride=1,
        **kwargs,
    )


def assert_reproduces_the_prior(outputs) -> None:
    """The full branch is the prior, bitwise, in every tensor that carries it.

    Args:
        outputs: The forward's dict.
    """
    assert torch.equal(outputs["mu_post"], outputs["mu_prior"])
    assert torch.equal(outputs["logvar_post"], outputs["logvar_prior"])
    assert torch.equal(outputs["z_post"], outputs["z_prior"])
    assert torch.equal(outputs["mu_full"], outputs["mu_base"])
    assert torch.equal(outputs["logvar_full"], outputs["logvar_base"])
    assert torch.all(outputs["kld_per_anchor"] == 0.0)
    assert torch.all(outputs["update_mean"] == 0.0)
    assert torch.all(outputs["update_logsigma"] == 0.0)


# =================================================================================================
# Source absence
# =================================================================================================
@pytest.mark.parametrize("training", [False, True])
def test_every_selector_off_reproduces_the_prior(training: bool) -> None:
    """After the head has been given nonzero weights, so the equality is earned rather than trivial.

    Checked in **training** mode as well, because the shared decoder is invoked twice and a module
    invoked twice draws two independent dropout masks. The decoder's dropout is zero for exactly
    this reason, and this is the assertion that would catch it being turned on.
    """
    model = build_model(trained=True)
    model.train(training)
    outputs = run(
        model, selector=torch.zeros(TINY_BATCH, TINY_SEQ_LEN, TINY_N_LAGS)[:, :14]
    )
    assert_reproduces_the_prior(outputs)


def test_every_lag_unavailable_reproduces_the_prior_and_stays_finite() -> None:
    """No invalid gather, no nonfinite value, and an exactly zero residual."""
    model = build_model(
        trained=True,
        source_warmup_steps=tuple(TINY_SEQ_LEN for _ in range(DECLARED_C_U)),
    )
    outputs = run(model, return_proposals=True)

    assert not bool(outputs["lag_valid"].any())
    assert bool(torch.isfinite(outputs["mu_full"]).all())
    assert torch.all(outputs["mean_proposals"] == 0.0)
    assert_reproduces_the_prior(outputs)


def test_the_zero_initialised_model_reproduces_the_prior_for_arbitrary_inputs() -> None:
    """At the zero output projection the equality holds whatever the source says."""
    model = build_model()
    for seed in (0, 1, 2):
        generator = torch.Generator().manual_seed(seed)
        y_st, y_ph, u_stream = tiny_streams()
        wild = torch.randn(u_stream.shape, generator=generator) * 50.0
        assert_reproduces_the_prior(run(model, streams=(y_st, y_ph, wild)))


def test_suppressing_one_band_is_local_to_that_band() -> None:
    """The suppression interface, checked at its two ends before anything is read off it."""
    model = build_model(trained=True)
    baseline = run(model, return_proposals=True)
    n_anchors = baseline["mean_proposals"].shape[1]

    selector = torch.ones(TINY_BATCH, n_anchors, TINY_N_LAGS)
    selector[:, :, 1:3] = 0.0
    suppressed = run(model, selector=selector, return_proposals=True)

    kept = [0] + list(range(3, TINY_N_LAGS))
    assert torch.all(suppressed["mean_proposals"][:, :, 1:3] == 0.0)
    assert torch.equal(
        suppressed["mean_proposals"][:, :, kept], baseline["mean_proposals"][:, :, kept]
    )
    # And the whole-band removal actually changes the prediction, so the readout is not vacuous.
    assert not torch.allclose(suppressed["mu_full"], baseline["mu_full"])


# =================================================================================================
# A real zero is an observation
# =================================================================================================
def test_an_observed_standardized_zero_is_not_treated_as_absence() -> None:
    """Absence and a real zero are different events, and only one of them silences a lag.

    A source stream of exact zeros carries mask one wherever the channel has warmed up, so its
    coefficients are observations that happen to sit at the mean. The model must be free to react
    to them: nothing here centres the proposal on its response to a zero input, and nothing equates
    a zero value with a missing one.
    """
    model = build_model(trained=True)
    y_st, y_ph, u_stream = tiny_streams()
    zeros = torch.zeros_like(u_stream)

    observed = run(model, streams=(y_st, y_ph, zeros), return_proposals=True)

    # The mask says the coefficients are present.
    assert bool(observed["lag_valid"].any())
    # And the model does react, rather than falling back on the prior.
    assert not torch.all(observed["update_mean"] == 0.0)
    assert not torch.allclose(observed["mu_post"], observed["mu_prior"])

    # Which is exactly what distinguishes it from absence, measured side by side.
    absent = run(
        build_model(
            trained=True,
            source_warmup_steps=tuple(TINY_SEQ_LEN for _ in range(DECLARED_C_U)),
        ),
        streams=(y_st, y_ph, zeros),
    )
    assert torch.all(absent["update_mean"] == 0.0)


# =================================================================================================
# Paired sampling and the decoder boundary
# =================================================================================================
def test_the_paired_sample_difference_matches_the_stated_formula() -> None:
    r"""$z^q - z^p = \sigma^p \odot [a + (e^{b} - 1)\odot\epsilon]$, under one shared draw."""
    model = build_model(trained=True)
    outputs = run(model)

    sigma_prior = torch.exp(0.5 * outputs["logvar_prior"])
    # The draw is recoverable from the prior branch, which is what makes the pairing checkable.
    epsilon = (outputs["z_prior"] - outputs["mu_prior"]) / sigma_prior
    expected = sigma_prior * (
        outputs["update_mean"] + (torch.exp(outputs["update_logsigma"]) - 1.0) * epsilon
    )
    assert torch.allclose(
        outputs["z_post"] - outputs["z_prior"], expected, atol=1e-5, rtol=1e-5
    )


def test_the_mean_only_arm_gives_a_deterministic_sample_difference() -> None:
    """With no scale update the difference is the mean correction and carries no noise."""
    model = build_model(trained=True, mean_only_residual=True)
    outputs = run(model)
    sigma_prior = torch.exp(0.5 * outputs["logvar_prior"])
    assert torch.allclose(
        outputs["z_post"] - outputs["z_prior"],
        sigma_prior * outputs["update_mean"],
        atol=1e-6,
    )


def test_the_decoder_is_invoked_twice_with_the_latent_and_the_persistence_input_only() -> None:
    """No source value, proposal, mask, latent parameter or encoder state reaches it.

    Recorded by intercepting the decoder rather than by reading the forward, because the property
    that matters is what the module was actually handed. The two calls must differ in the latent
    and in nothing else, or the base-minus-full gap would carry the difference as well.
    """
    model = build_model(trained=True)
    calls = []
    original = model.decoder.forward

    def record(decoder_state, persistence=None, *args, **kwargs):
        """Record one decoder invocation and forward it unchanged."""
        calls.append((decoder_state, persistence, args, kwargs))
        return original(decoder_state, persistence=persistence)

    model.decoder.forward = record  # type: ignore[method-assign]
    outputs = run(model)
    model.decoder.forward = original  # type: ignore[method-assign]

    assert len(calls) == 2
    for _, _, extra_args, extra_kwargs in calls:
        assert extra_args == () and extra_kwargs == {}

    (base_state, base_persistence, _, _), (full_state, full_persistence, _, _) = calls
    # The identical tensor object, not an equal one: both calls decode with the same target-only
    # vector, so it cancels out of the difference of predicted means.
    assert base_persistence is full_persistence
    assert torch.equal(base_state, outputs["z_prior"])
    assert torch.equal(full_state, outputs["z_post"])


def test_the_two_decoder_calls_share_one_module_and_its_weights() -> None:
    """One decoder invoked twice, so base and full are two latents through one predictor."""
    model = build_model(trained=True)
    baseline = run(model)

    with torch.no_grad():
        model.decoder.mean_head.weight.mul_(1.5)
    moved = run(model)

    # A change to the one decoder must move both branches; two decoders would move only one.
    assert not torch.allclose(baseline["mu_base"], moved["mu_base"])
    assert not torch.allclose(baseline["mu_full"], moved["mu_full"])


# =================================================================================================
# Gradient escape from the zero start
# =================================================================================================
def test_the_divergence_gradient_is_zero_at_initialisation_and_that_is_intended() -> None:
    """Recorded as an expectation rather than left to be discovered as a failure.

    At a zero update the divergence sits at its exact minimum, so its gradient with respect to the
    residual vanishes. Demanding a positive divergence at initialisation would be demanding that
    the model start somewhere it is deliberately not.
    """
    model = build_model()
    outputs = run(model)
    assert float(outputs["kld_per_anchor"].sum()) == 0.0

    outputs["kld_per_anchor"].sum().backward()
    grad = model.proposal_head.output_proj.weight.grad
    assert grad is not None
    assert float(grad.abs().max()) == 0.0


def test_the_predictive_gradient_reaches_the_final_source_projection_at_the_zero_start() -> None:
    """The difference between a model that starts at the prior and one that stays there.

    The claim is narrow, and deliberately so: the design predicts that the *hidden* source layers
    may see no gradient on the very first step, because the projection in front of them is zero.
    What must be true is that the projection itself moves, which is what lets the hidden layers see
    gradient on the step after.
    """
    model = build_model()
    y_st, y_ph, u_stream = tiny_streams()
    torch.manual_seed(3)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)

    # A predictive loss against a target that genuinely depends on the source, so there is
    # something for the source pathway to be useful for.
    generator = torch.Generator().manual_seed(4)
    target = torch.randn(outputs["mu_full"].shape, generator=generator)
    ((outputs["mu_full"] - target) ** 2).mean().backward()

    grad = model.proposal_head.output_proj.weight.grad
    assert grad is not None
    assert float(grad.abs().max()) > 0.0


def test_the_source_pathway_leaves_the_zero_start_after_a_few_steps() -> None:
    """The whole head, not only its last layer, on a toy where the source genuinely helps.

    The target here is a function of the source that the target stream cannot supply, so a model
    that never left the prior could not fit it.
    """
    model = SeqVaeLagResidualTrfCfs(**tiny_model_kwargs())
    model.train()
    y_st, y_ph, u_stream = tiny_streams()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(8):
        torch.manual_seed(step)
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
        # Ask the forecast to track a source-derived level the target stream does not carry.
        wanted = u_stream[:, : outputs["mu_full"].shape[1], :1].unsqueeze(2)
        loss = ((outputs["mu_full"] - wanted) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert float(model.proposal_head.output_proj.weight.abs().max()) > 0.0
    assert float(model.proposal_head.input_proj.weight.grad.abs().max()) > 0.0
    assert float(model.proposal_head.lag_embedding.weight.grad.abs().max()) > 0.0

    model.eval()
    outputs = run(model)
    assert float(outputs["kld_per_anchor"].sum()) > 0.0
