r"""Which DDP strategy the configured parameter usage permits, and the evidence that it is safe.

Plain ``'ddp'`` means ``find_unused_parameters=False``: the reducer expects every parameter to
be marked ready in every backward, and one that is not raises or deadlocks on a real multi-GPU
box. The strategy string is a *claim* about the model; the grad-coverage tests at the bottom are
the *evidence* -- without them this file would only assert that a function returns the string it
was written to return.

This model has exactly one config-decided starvation source: the decoder log-variance heads are
consumed only under ``likelihood: gaussian_nll``. The sibling's other source -- the attention
output projection under a head-structured latent -- does not exist as a config axis here,
because the net freezes ``W_o`` unconditionally and a frozen parameter is outside the reducer's
expectation set.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_rws.tests.conftest import make_stub_batch
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config; ``setup_config`` is never called."""
    driver = LagAttnRwsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    return driver


def _config(**vae_overrides) -> dict:
    """A minimal config carrying only the keys the strategy selector reads."""
    return {"model_config": {"VAE_model": dict(vae_overrides)}}


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
def test_the_shipped_config_earns_every_parameter_reachable(trainer):
    """The payoff of the learned observation variance plus the unconditional W_o freeze: the
    reducer can expect every parameter, which is what the ``'ddp'`` shorthand used to say."""
    assert trainer.ddp_kwargs(trainer.config)["find_unused_parameters"] is False


def test_a_single_device_needs_no_strategy(trainer):
    assert trainer.select_ddp_strategy(1, trainer.config) == "auto"


def test_an_mse_likelihood_starves_the_logvar_heads(trainer):
    """The tiny smoke configuration: mse stops consuming the decoder log-variance heads."""
    assert trainer.ddp_kwargs(_config(likelihood="mse"))["find_unused_parameters"] is True


def test_the_buffer_broadcast_is_off_and_the_gradients_are_bucket_views(trainer):
    """Two performance settings the shorthand strings cannot express, which is why the selector
    returns an instance. ``broadcast_buffers=False`` is safe only because every buffer in this
    family is a deterministic function of the config -- filter banks, rotary tables, causal masks,
    index grids -- and there is no ``BatchNorm`` anywhere to carry a genuinely per-rank running
    statistic."""
    kwargs = trainer.ddp_kwargs(trainer.config)

    assert kwargs["broadcast_buffers"] is False
    assert kwargs["gradient_as_bucket_view"] is True


def test_static_graph_is_not_claimed(trainer):
    """The negative control on the performance bundle, and a correctness call rather than an
    omission: the loss-spike breaker substitutes a zero-weighted sum over every parameter on a
    skipped batch, which is a structurally different backward from the one iteration 1 recorded.
    ``static_graph=True`` promises DDP that never happens, and the breaker ships enabled."""
    assert "static_graph" not in trainer.ddp_kwargs(trainer.config)


def test_the_settings_reach_the_strategy_object(trainer):
    """``DDPStrategy`` forwards unrecognised kwargs into ``_ddp_kwargs`` and on to
    ``DistributedDataParallel``. That name is Lightning-internal, so it is asserted **here only**
    -- everything else in this file reads :meth:`ddp_kwargs`, which is ours. If Lightning ever
    stops forwarding them, this is the test that says so rather than a silently slower run."""
    strategy = trainer.select_ddp_strategy(8, trainer.config)

    assert type(strategy).__name__ == "DDPStrategy"
    assert strategy._ddp_kwargs == trainer.ddp_kwargs(trainer.config)


def test_the_selector_is_a_pure_function_of_config(trainer):
    """The framework passes the *Lightning module* as ``model``, not the raw net; a selector
    that read a net attribute off it would find nothing and silently regress the shipped config
    to the slow strategy on the one box where it costs."""
    without_model = trainer.select_ddp_strategy(8, trainer.config)
    with_wrapper = trainer.select_ddp_strategy(8, trainer.config, model=object())

    assert without_model._ddp_kwargs == with_wrapper._ddp_kwargs


def test_the_hook_is_the_un_prefixed_name_the_framework_looks_up():
    """The framework calls ``select_ddp_strategy`` and nothing else; an underscore-prefixed
    override would never run and every multi-device run would fall back to the base's plain
    'ddp' -- including the mse configuration that must not use it."""
    assert "select_ddp_strategy" in vars(LagAttnRwsTrainer)
    assert "_select_ddp_strategy" not in vars(LagAttnRwsTrainer)


def test_the_override_reaches_the_trainer_kwargs(trainer, monkeypatch):
    """``_build_trainer_kwargs`` sets ``strategy`` only under CUDA, so the patch is what makes
    this assertion mean anything on a CPU box. Asserted on the kwargs dict, never on a real
    ``Trainer`` -- the shipped config names seven CUDA devices."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.cuda_devices = [0, 1, 2, 3, 4, 5, 6]

    kwargs = trainer._build_trainer_kwargs([])

    assert type(kwargs["strategy"]).__name__ == "DDPStrategy"
    assert kwargs["accelerator"] == "gpu"


def test_no_strategy_key_is_set_on_a_cpu_box(trainer, monkeypatch):
    """Documents why every test above calls the hook directly instead of reading the kwargs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    kwargs = trainer._build_trainer_kwargs([])

    assert "strategy" not in kwargs
    assert kwargs["accelerator"] == "cpu"


# --------------------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------------------
def _starved_parameters(module, batch_idx: int) -> list[str]:
    """Backward one training step and name the trainable parameters left without a gradient."""
    module.zero_grad(set_to_none=True)
    loss, _ = module.compute_loss_and_metrics(make_stub_batch(4), batch_idx, "train")
    loss.backward()
    return [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, beta_prior
):
    """What actually licenses ``find_unused_parameters=False`` for the shipped config.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step.
    Both anchor weights, because the prior scale rate is the one term a config can switch on:
    the coverage claim must hold for the objective production actually optimises.
    """
    module = task(hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under plain 'ddp' "
        f"the reducer raises on exactly these."
    )


@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_under_mse_the_decoder_logvar_head_is_what_starves(task, perturb_posterior, beta_prior):
    """The mirror image, and the justification for the fallback strategy: with mse the decoder
    log-variance head is trainable and unused. If this ever stops holding, the likelihood axis
    of the strategy selector is unnecessary and should be deleted rather than kept. The anchor
    weight must not change the starved set either way: the prior's log-variance head sits
    outside it under both, reached through the KL and the base forecast."""
    module = task(hparams={"likelihood": "mse", "beta_prior": beta_prior})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert starved, "no parameter starved under mse; the fallback strategy is unjustified"
    assert all("logvar_head" in name for name in starved), (
        f"unexpected starvation beyond the decoder logvar head: {starved}"
    )


def test_the_attention_projection_is_frozen_out_of_the_expectation_set(task):
    """The mechanism that removes the sibling's second starvation axis: frozen means not
    expected, not merely unused."""
    module = task()

    assert not any(
        parameter.requires_grad
        for parameter in module.orig_model.lag_attn.W_o.parameters()
    )
