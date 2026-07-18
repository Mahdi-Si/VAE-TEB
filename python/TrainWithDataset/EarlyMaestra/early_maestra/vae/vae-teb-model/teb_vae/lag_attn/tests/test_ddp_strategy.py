r"""Which DDP strategy the configured parameter usage permits, and the evidence that it is safe.

Plain ``'ddp'`` means ``find_unused_parameters=False``: the reducer expects every parameter to be
marked ready in every backward, and a parameter that is not raises or deadlocks on a real
multi-GPU box. ``ddp_find_unused_parameters_true`` is the safe fallback and costs a full extra
traversal of the autograd graph every step.

The strategy string is a *claim* about the model. The grad-coverage tests at the bottom are the
*evidence*; without them this file would only assert that a function returns the string it was
written to return.

None of this can be tested against a real process group here, and it does not need to be: the
selection is a pure function of config, and the framework's own suite already proves the hook is
what ``build_trainer`` calls. The one place those two facts meet -- the strategy actually reaching
the ``Trainer`` kwargs -- needs CUDA to be visible, so that test monkeypatches
``torch.cuda.is_available``. Without the patch the accelerator branch never runs and every
assertion about ``strategy`` passes vacuously.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.tests.conftest import make_stub_batch

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config.

    Constructed directly rather than through the framework's ``make_graph_model`` helper, which
    builds its own stub subclass and so cannot exercise this class's override. Nothing here calls
    ``setup_config``, so the shipped config's Linux output path is never touched -- only stored.
    """
    from teb_vae.lag_attn.trainer import LagAttnTrainer

    driver = LagAttnTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    return driver


def _config(**vae_overrides) -> dict:
    """A minimal config carrying only the keys the strategy selector reads."""
    return {"model_config": {"VAE_model": dict(vae_overrides)}}


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
def test_the_shipped_config_earns_plain_ddp(trainer):
    """The payoff of learned observation variance and the freeze flag together."""
    assert trainer.select_ddp_strategy(8, trainer.config) == "ddp"


def test_a_single_device_needs_no_strategy(trainer):
    assert trainer.select_ddp_strategy(1, trainer.config) == "auto"


def test_a_fixed_sigma_obs_starves_the_logvar_heads(trainer):
    """A debug run at fixed variance stops consuming the decoder log-variance heads."""
    config = _config(
        likelihood="gaussian_nll", sigma_obs=1.0, head_structured_latent=True,
        freeze_unused_attn_proj=True,
    )

    assert trainer.select_ddp_strategy(8, config) == "ddp_find_unused_parameters_true"


def test_an_mse_likelihood_starves_them_too(trainer):
    config = _config(
        likelihood="mse", sigma_obs="learned", head_structured_latent=True,
        freeze_unused_attn_proj=True,
    )

    assert trainer.select_ddp_strategy(8, config) == "ddp_find_unused_parameters_true"


def test_an_unfrozen_projection_under_a_head_structured_latent_starves_the_projection(trainer):
    """The case the freeze flag exists for."""
    config = _config(
        likelihood="gaussian_nll", sigma_obs="learned", head_structured_latent=True,
        freeze_unused_attn_proj=False,
    )

    assert trainer.select_ddp_strategy(8, config) == "ddp_find_unused_parameters_true"


def test_a_flat_latent_leaves_the_projection_consumed_whether_frozen_or_not(trainer):
    """Without head structure the posterior consumes the projection's output, so nothing starves.

    This is why the selector ANDs the two flags rather than reading the freeze flag alone.
    """
    config = _config(
        likelihood="gaussian_nll", sigma_obs="learned", head_structured_latent=False,
        freeze_unused_attn_proj=False,
    )

    assert trainer.select_ddp_strategy(8, config) == "ddp"


def test_the_selector_ignores_the_model_argument(trainer):
    """The trap this signature exists to avoid.

    ``build_trainer`` passes the *Lightning module*, not the raw net. A selector that read
    ``model.frozen_attn_proj`` would find nothing on the wrapper, conclude the projection was
    starved, and silently regress the shipped config to the slow strategy -- on a multi-GPU box
    only, where it costs performance and nothing fails.
    """
    without_model = trainer.select_ddp_strategy(8, trainer.config)
    with_wrapper = trainer.select_ddp_strategy(8, trainer.config, model=object())

    assert without_model == with_wrapper == "ddp"


def test_the_hook_is_the_un_prefixed_name_the_framework_looks_up():
    """Carrying the old underscore-prefixed name over would be a silent no-op.

    The framework looks up ``select_ddp_strategy`` and nothing else, so a ``_select_ddp_strategy``
    would never be called and the run would fall back to the base's default -- which returns plain
    ``'ddp'`` for any multi-device run, including the configurations above that must not use it.
    """
    from teb_vae.lag_attn.trainer import LagAttnTrainer

    assert "select_ddp_strategy" in vars(LagAttnTrainer)
    assert "_select_ddp_strategy" not in vars(LagAttnTrainer)


def test_the_override_reaches_the_trainer_kwargs(trainer, monkeypatch):
    """The hook and the builder, joined.

    ``_build_trainer_kwargs`` sets ``strategy`` only under CUDA, so this patch is what makes the
    assertion mean anything on a CPU box.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7]

    kwargs = trainer._build_trainer_kwargs([])

    assert kwargs["strategy"] == "ddp"
    assert kwargs["accelerator"] == "gpu"


def test_no_strategy_key_is_set_on_a_cpu_box(trainer, monkeypatch):
    """Documents why every test above calls the hook directly instead of reading the kwargs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    kwargs = trainer._build_trainer_kwargs([])

    assert "strategy" not in kwargs
    assert kwargs["accelerator"] == "cpu"


# --------------------------------------------------------------------------------------
# sync_batchnorm, which the port deliberately changed
# --------------------------------------------------------------------------------------
def test_sync_batchnorm_follows_the_config_on_a_multi_device_run(trainer, monkeypatch):
    """A deliberate behaviour change, asserted at 8 devices because 1 would be vacuous.

    The trainer this was ported from hardcoded ``len(cuda_devices) > 1`` and so ran with
    ``sync_batchnorm=True`` on the prod box while its own config said ``false``. The framework ANDs
    the config with the device count, so the config now wins.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7]
    assert trainer.config["advanced_config"]["trainer"]["sync_batchnorm"] is False

    assert trainer._build_trainer_kwargs([])["sync_batchnorm"] is False

    trainer.config["advanced_config"]["trainer"]["sync_batchnorm"] = True
    assert trainer._build_trainer_kwargs([])["sync_batchnorm"] is True


def test_sync_batchnorm_is_off_on_one_device_whatever_the_config_says(trainer, monkeypatch):
    """SyncBatchNorm's forward needs an initialised process group; one device has none."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.cuda_devices = [0]
    trainer.config["advanced_config"]["trainer"]["sync_batchnorm"] = True

    assert trainer._build_trainer_kwargs([])["sync_batchnorm"] is False


# --------------------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("batch_idx", [0, 1], ids=["perm-step", "plain-step"])
def test_no_parameter_is_left_without_a_gradient(task, shipped_kwargs, perturb_posterior, batch_idx):
    """What actually licenses ``find_unused_parameters=False``, at the shipped flag set.

    Run against ``shipped_kwargs`` rather than the smaller fixture set: head-structured latents and
    the freeze flag are precisely the flags that decide whether a parameter starves, and a set that
    leaves them off cannot see the failure.
    """
    module = task(model_kwargs=shipped_kwargs)
    perturb_posterior(module.orig_model)

    module.zero_grad(set_to_none=True)
    loss, _ = module.compute_loss_and_metrics(make_stub_batch(), batch_idx, "train")
    loss.backward()

    starved = [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not starved, (
        f"parameters expecting a gradient but not receiving one on batch_idx={batch_idx}: "
        f"{starved}. Under plain 'ddp' the reducer raises on exactly these."
    )


def test_freezing_removes_the_projection_from_the_expectation_set(task, shipped_kwargs):
    """The mechanism behind the strategy choice: frozen means not expected, not merely unused."""
    module = task(model_kwargs=shipped_kwargs)

    assert module.orig_model.frozen_attn_proj is True
    assert not any(
        parameter.requires_grad for parameter in module.orig_model.lag_attn.W_o.parameters()
    )


def test_the_unfrozen_projection_is_what_would_starve(task, shipped_kwargs, perturb_posterior):
    """The mirror image, and the justification for the fallback strategy.

    With the freeze off, the projection is trainable and still unused -- so it receives no
    gradient, and this is the configuration that genuinely needs find_unused_parameters.
    """
    module = task(model_kwargs=dict(shipped_kwargs, freeze_unused_attn_proj=False))
    perturb_posterior(module.orig_model)

    module.zero_grad(set_to_none=True)
    loss, _ = module.compute_loss_and_metrics(make_stub_batch(), 1, "train")
    loss.backward()

    starved = [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert any("W_o" in name for name in starved), (
        "the unfrozen projection received a gradient; if that is now true, the whole "
        "freeze/strategy dance is unnecessary and should be deleted rather than tested"
    )


def test_freezing_the_projection_is_a_forward_no_op(task, shipped_kwargs, inputs):
    """Numerically free, which is what makes it an acceptable price for the fast strategy."""
    frozen = task(model_kwargs=shipped_kwargs)
    trainable = task(model_kwargs=dict(shipped_kwargs, freeze_unused_attn_proj=False))
    trainable.orig_model.load_state_dict(frozen.orig_model.state_dict())
    frozen.orig_model.eval()
    trainable.orig_model.eval()

    torch.manual_seed(7)
    reference = frozen.orig_model(*inputs)
    torch.manual_seed(7)
    got = trainable.orig_model(*inputs)

    for key in ("mu_prior", "mu_post", "mu_full", "te_lag_map"):
        assert torch.allclose(reference[key], got[key], atol=1e-6), f"drift on {key}"
