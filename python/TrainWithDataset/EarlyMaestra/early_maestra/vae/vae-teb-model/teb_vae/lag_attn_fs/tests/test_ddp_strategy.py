r"""Which DDP strategy the configured parameter usage permits, and the evidence that it is safe.

``find_unused_parameters=False`` makes the reducer expect every parameter to be marked ready in
every backward, and one that is not raises or deadlocks on a real multi-GPU box. The selector is a
*claim* about the model; the grad-coverage tests at the bottom are the *evidence* -- without them
this file would only assert that a function returns what it was written to return.

The claim is inherited, and the question this file asks is whether it survives the target-domain
change. It has one config-decided starvation source: the decoder log-variance heads are consumed
only under ``likelihood: gaussian_nll``. Those heads are the tensors this model *widened*, from
$16$ outputs to $C_{\mathrm{keep}} = 78$, so if the change were going to move the starved set this
is where it would show.

``broadcast_buffers=False`` needs its justification restated rather than inherited, and the
restatement is not the one the design document predicted. That buffer list includes the raw-target
index grid, and this model **does** still carry it: the base constructor registers it and a
subclass can only drop it by overriding ``__init__``, which the width hook exists to avoid. It is
simply never read. The setting is safe for the reason it was always safe -- every buffer here is a
deterministic function of the config, and there is no ``BatchNorm`` anywhere to carry a genuinely
per-rank running statistic -- and an unread buffer only makes the broadcast more wasteful.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

from .conftest import make_stub_batch

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config; ``setup_config`` is never called."""
    driver = LagAttnFsTrainer(config_file_path=str(_CONFIG))
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
    reducer can expect every parameter."""
    assert trainer.ddp_kwargs(trainer.config)["find_unused_parameters"] is False


def test_a_single_device_needs_no_strategy(trainer):
    assert trainer.select_ddp_strategy(1, trainer.config) == "auto"


def test_the_smoke_configs_mse_selects_the_fallback(trainer):
    """``tiny.yaml`` ships ``likelihood: mse`` precisely so the smoke path exercises this branch
    where it is cheap to observe, rather than leaving it configured and never run."""
    from teb_vae.lag_attn.config import load_config

    tiny = load_config(str(_TINY))

    assert tiny["model_config"]["VAE_model"]["likelihood"] == "mse"
    assert trainer.ddp_kwargs(tiny)["find_unused_parameters"] is True
    assert trainer.ddp_kwargs(_config(likelihood="mse"))["find_unused_parameters"] is True


def test_the_buffer_broadcast_is_off_and_the_gradients_are_bucket_views(trainer):
    """Two performance settings the shorthand strategy strings cannot express, which is why the
    selector returns an instance."""
    kwargs = trainer.ddp_kwargs(trainer.config)

    assert kwargs["broadcast_buffers"] is False
    assert kwargs["gradient_as_bucket_view"] is True


def test_no_buffer_is_a_running_statistic_so_the_broadcast_is_safe_to_skip(shipped_gated):
    """What licenses ``broadcast_buffers=False``: every buffer is a deterministic function of the
    config, built identically in each rank's constructor, so the broadcast restores values that
    were never going to differ. A ``BatchNorm`` running statistic is the one kind that genuinely
    diverges per rank, and there is none.

    The raw-target index grid is still among those buffers -- inherited, non-persistent, and never
    read by this model -- which makes the broadcast marginally more wasteful and changes nothing
    about whether it is safe to skip."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**shipped_gated)

    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                   for module in model.modules())
    assert "future_index" in dict(model.named_buffers())
    assert "future_index" not in model.state_dict()  # non-persistent: it reaches no checkpoint


def test_static_graph_is_not_claimed(trainer):
    """A correctness call rather than an omission: the loss-spike breaker substitutes a
    zero-weighted sum over every parameter on a skipped batch, which is a structurally different
    backward from the one iteration 1 recorded. ``static_graph=True`` promises DDP that never
    happens, and the breaker ships enabled."""
    assert "static_graph" not in trainer.ddp_kwargs(trainer.config)


def test_the_settings_reach_the_strategy_object(trainer):
    """``DDPStrategy`` forwards unrecognised kwargs into ``_ddp_kwargs`` and on to
    ``DistributedDataParallel``. That name is Lightning-internal, so it is asserted here only."""
    strategy = trainer.select_ddp_strategy(8, trainer.config)

    assert type(strategy).__name__ == "DDPStrategy"
    assert strategy._ddp_kwargs == trainer.ddp_kwargs(trainer.config)


def test_the_selector_is_a_pure_function_of_config(trainer):
    """The framework passes the *Lightning module* as ``model``, not the raw net; a selector that
    read a net attribute off it would find nothing and silently regress the shipped config to the
    slow strategy on the one box where it costs."""
    without_model = trainer.select_ddp_strategy(8, trainer.config)
    with_wrapper = trainer.select_ddp_strategy(8, trainer.config, model=object())

    assert without_model._ddp_kwargs == with_wrapper._ddp_kwargs


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


@pytest.mark.parametrize("beta_prior", [0.0, 0.5], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, beta_prior
):
    """What actually licenses ``find_unused_parameters=False`` for the shipped config, re-earned on
    the widened decoder head.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step. Both
    anchor weights, at this model's own retuned value, because the prior scale rate is the one term
    a config can switch on.
    """
    module = task(hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under "
        f"find_unused_parameters=False the reducer raises on exactly these."
    )


@pytest.mark.parametrize("beta_prior", [0.0, 0.5], ids=["unanchored", "anchored"])
def test_under_mse_the_starved_set_is_exactly_the_decoder_logvar_head(
    task, perturb_posterior, beta_prior
):
    """The mirror image, and the justification for the fallback strategy: with mse the decoder
    log-variance head is trainable and unused. **Exactly** that head and nothing else -- if some
    other parameter starved here, ``find_unused_parameters=True`` would be covering for a second
    defect rather than for a documented configuration choice."""
    module = task(hparams={"likelihood": "mse", "beta_prior": beta_prior})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert set(starved) == {"decoder.logvar_head.weight", "decoder.logvar_head.bias"}, starved
    # And it is the widened head: this is the tensor whose shape the target domain changed.
    assert module.orig_model.decoder.logvar_head.bias.numel() == module.orig_model.c_y


def test_the_attention_projection_is_frozen_out_of_the_expectation_set(task):
    """The mechanism that removes the second starvation axis: frozen means not expected, not merely
    unused."""
    module = task()

    assert not any(
        parameter.requires_grad
        for parameter in module.orig_model.lag_attn.W_o.parameters()
    )


def test_the_hook_is_the_un_prefixed_name_the_framework_looks_up():
    """The framework calls ``select_ddp_strategy`` and nothing else. Inherited here, so what is
    asserted is that this driver did not shadow it with an underscore-prefixed copy that would
    never run."""
    from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

    assert "select_ddp_strategy" in vars(LagAttnRwsTrainer)
    assert "_select_ddp_strategy" not in vars(LagAttnFsTrainer)
    assert LagAttnFsTrainer.select_ddp_strategy is LagAttnRwsTrainer.select_ddp_strategy
