r"""Which DDP strategy the configured parameter usage permits, and the evidence that it is safe.

Plain ``'ddp'`` means ``find_unused_parameters=False``: the reducer expects every parameter to be
marked ready in every backward, and one that is not raises or deadlocks on a real multi-GPU box.
The strategy string is a *claim* about the model; the grad-coverage tests at the bottom are the
*evidence* -- without them this file would only assert that an inherited function returns the string
it was written to return.

The selector is inherited unchanged and keys on ``likelihood`` alone, which is the right axis here
for the same reasons it is there: the decoder log-variance heads are consumed only under
``gaussian_nll``, and the attention output projection ``W_o`` -- the other candidate -- is frozen
unconditionally by the constructor, so it is outside the reducer's expectation set entirely.

What is new is the front ends, and they are the reason this file exists in this package rather than
being taken as read. Two of their properties are exactly what a plain-``ddp`` run needs and neither
is obvious from the strategy string: the featurisation's masking is multiplicative and
unconditional, so no parameter drops out of the graph on a batch with a gap; and the stage
projections carry biases, so even a **fully invalid** window -- which featurises to an exactly zero
vector -- still puts gradient on every stage. Both are asserted, on real batches, below.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer

from .conftest import BATCH, SEQ_LEN, make_stub_batch

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def driver(tmp_path):
    """A driver on the shipped config; ``setup_config`` is never called."""
    instance = LagAttnTrfE2ETrainer(config_file_path=str(_CONFIG))
    instance.output_base_dir = str(tmp_path)
    instance.train_results_dir = str(tmp_path / "train_results")
    return instance


def _config(**vae_overrides) -> dict:
    """A minimal config carrying only the keys the strategy selector reads."""
    return {"model_config": {"VAE_model": dict(vae_overrides)}}


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


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
def test_the_shipped_config_earns_every_parameter_reachable(driver):
    """The payoff of the learned observation variance plus the unconditional W_o freeze."""
    assert driver.ddp_kwargs(driver.config)["find_unused_parameters"] is False


def test_a_single_device_needs_no_strategy(driver):
    assert driver.select_ddp_strategy(1, driver.config) == "auto"


def test_an_mse_likelihood_starves_the_logvar_heads(driver):
    """The tiny smoke configuration: mse stops consuming the decoder log-variance heads."""
    assert driver.ddp_kwargs(_config(likelihood="mse"))["find_unused_parameters"] is True


def test_the_buffer_broadcast_is_off(driver):
    """This architecture is the one with the most to gain and the most to check: its buffers are
    the eight fixed anti-alias filter banks on top of the rotary tables, causal masks and raw-target
    index grid the siblings carry -- $1.5$ MiB re-broadcast from rank 0 on every forward. All are
    deterministic functions of the config, and there is no ``BatchNorm`` here to carry a per-rank
    running statistic, so the broadcast only ever restored values that could not have differed."""
    assert driver.ddp_kwargs(driver.config)["broadcast_buffers"] is False


def test_the_selector_is_a_pure_function_of_config(driver):
    """The framework passes the *Lightning module* as ``model``, not the raw net; a selector that
    read a net attribute off it would find nothing and silently regress the shipped config to the
    slow strategy on the one box where it costs."""
    without_model = driver.select_ddp_strategy(7, driver.config)
    with_wrapper = driver.select_ddp_strategy(7, driver.config, model=object())

    assert without_model._ddp_kwargs == with_wrapper._ddp_kwargs


def test_the_selector_is_inherited_rather_than_re_pointed():
    """Nothing about the input representation changes which parameters a backward reaches, so an
    override here would be a second copy of a decision that has to stay the same in both packages
    for the two runs to be comparable at all."""
    assert "select_ddp_strategy" not in vars(LagAttnTrfE2ETrainer)


def test_the_override_reaches_the_trainer_kwargs(driver, monkeypatch):
    """``_build_trainer_kwargs`` sets ``strategy`` only under CUDA, so the patch is what makes this
    assertion mean anything on a CPU box. Asserted on the kwargs dict, never on a real ``Trainer``
    -- the shipped config names seven CUDA devices."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    driver.cuda_devices = [0, 1, 2, 3, 4, 5, 6]

    kwargs = driver._build_trainer_kwargs([])

    assert type(kwargs["strategy"]).__name__ == "DDPStrategy"
    assert kwargs["accelerator"] == "gpu"


# --------------------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, stub_batch, beta_prior
):
    """What actually licenses ``find_unused_parameters=False`` for the shipped config.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step. Both
    anchor weights, because the prior scale rate is the one term a config can switch on: the
    coverage claim must hold for the objective production actually optimises.
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
    assert not any("frontend" in name for name in starved)


def test_every_front_end_parameter_receives_a_gradient_through_a_gap(
    task, perturb_posterior, stub_batch
):
    """The stub batch carries a planted weight gap, so this is the masked path rather than a
    uniformly valid one. The featurisation neutralises invalid samples by multiplying, never by
    branching, which is what keeps every parameter in the graph -- a ``if mask.any():`` optimisation
    would pass every unit test in the package and deadlock a seven-rank run."""
    module = task()
    perturb_posterior(module.orig_model)

    starved = set(_starved_parameters(module, stub_batch))

    front_end = [
        name
        for name, _ in module.orig_model.named_parameters()
        if name.startswith(("target_frontend.", "source_frontend."))
    ]
    assert front_end, "the model has no front-end parameters; this test is measuring nothing"
    assert [name for name in front_end if name in starved] == []


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
    assert [name for name in front_end if name in starved] == []


def test_the_attention_projection_is_frozen_out_of_the_expectation_set(task):
    """The mechanism that removes the second starvation axis: frozen means not expected, not
    merely unused."""
    module = task()

    assert not any(
        parameter.requires_grad
        for parameter in module.orig_model.lag_attn.W_o.parameters()
    )
