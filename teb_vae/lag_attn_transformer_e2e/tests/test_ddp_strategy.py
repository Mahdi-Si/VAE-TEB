r"""The evidence that the DDP strategy the configured likelihood selects is safe for this model.

Plain ``'ddp'`` means ``find_unused_parameters=False``: the reducer expects every parameter to be
marked ready in every backward. The selector is inherited unchanged, keys on ``likelihood`` alone,
and is tested where it is defined; the strategy string is only a *claim* about the model, and the
backward passes below are the evidence for it on this architecture.

What is new here is the front ends. The featurisation's masking is multiplicative and
unconditional, so no parameter drops out of the graph on a batch with a gap; and the stage
projections carry biases, so even a **fully invalid** window -- which featurises to an exactly zero
vector -- still puts gradient on every stage.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import BATCH, SEQ_LEN, make_stub_batch


def _starved_parameters(module, batch) -> list:
    """Backward one training step and name the trainable parameters left without a gradient.

    Args:
        module: The task.
        batch: The batch to step on.

    Returns:
        Parameter names, in ``named_parameters`` order.
    """
    module.zero_grad(set_to_none=True)
    loss, _ = module.compute_loss_and_metrics(batch, 0, "train")
    loss.backward()
    return [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, stub_batch, beta_prior
):
    """What licenses ``find_unused_parameters=False`` for the shipped likelihood, on the stub batch
    and its planted weight gap, so the masked path rather than a uniformly valid one.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight. Both anchor weights, because the prior scale rate is the one term a config
    can switch on.
    """
    module = task(hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, stub_batch)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under plain 'ddp' "
        f"the reducer raises on exactly these."
    )


def test_under_mse_the_decoder_logvar_head_is_what_starves(
    task, perturb_posterior, stub_batch
):
    """The mirror image, and the justification for the fallback strategy -- but also the assertion
    that would catch a **front-end** parameter that starves, since it names the starved set exactly
    rather than merely requiring it to be non-empty."""
    module = task(hparams={"likelihood": "mse"})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, stub_batch)

    assert starved, "no parameter starved under mse; the fallback strategy is unjustified"
    assert all("logvar_head" in name for name in starved), (
        f"unexpected starvation beyond the decoder logvar head: {starved}"
    )


def test_every_front_end_parameter_receives_a_gradient_on_a_fully_masked_batch(
    task, perturb_posterior
):
    """The extreme case, which the stage projections' biases exist for: a fully invalid window
    featurises to an exactly zero vector, and without a bias the whole cascade would emit zeros and
    the projections would receive none. It is also the case an "empty window, skip it" shortcut
    would have been written for."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(BATCH, SEQ_LEN)
    batch.weight = torch.zeros_like(batch.weight)

    starved = set(_starved_parameters(module, batch))

    front_end = [
        name
        for name, _ in module.orig_model.named_parameters()
        if name.startswith(("target_frontend.", "source_frontend."))
    ]
    assert front_end
    assert [name for name in front_end if name in starved] == []
