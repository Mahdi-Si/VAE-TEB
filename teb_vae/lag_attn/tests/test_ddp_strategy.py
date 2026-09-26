r"""Which DDP strategy the configured parameter usage permits, and the evidence that it is safe.

Plain ``'ddp'`` means ``find_unused_parameters=False``: the reducer expects every parameter to be
marked ready in every backward, and a parameter that is not raises or deadlocks on a real
multi-GPU box. ``ddp_find_unused_parameters_true`` is the safe fallback and costs a full extra
traversal of the autograd graph every step.

The strategy string is a *claim* about the model. The grad-coverage tests at the bottom are the
*evidence*; without them this file would only assert that a function returns the string it was
written to return.

None of this can be tested against a real process group here, and it does not need to be: the
selection is a pure function of config. The one place it meets the framework -- the strategy
actually reaching the ``Trainer`` kwargs -- needs CUDA to be visible, so that test monkeypatches
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
    assert trainer.select_ddp_strategy(1, trainer.config) == "auto"


_FUT = "ddp_find_unused_parameters_true"


@pytest.mark.parametrize(
    "likelihood, sigma_obs, head_structured_latent, freeze_unused_attn_proj, expected",
    [
        # A debug run at fixed variance stops consuming the decoder log-variance heads.
        ("gaussian_nll", 1.0, True, True, _FUT),
        ("mse", "learned", True, True, _FUT),
        # The case the freeze flag exists for.
        ("gaussian_nll", "learned", True, False, _FUT),
        # Without head structure the posterior consumes the projection's output, so nothing
        # starves: why the selector ANDs the two flags rather than reading the freeze flag alone.
        ("gaussian_nll", "learned", False, False, "ddp"),
    ],
    ids=["fixed-sigma", "mse", "unfrozen-head-structured", "flat-latent"],
)
def test_the_selector_falls_back_exactly_when_a_parameter_would_starve(
    trainer, likelihood, sigma_obs, head_structured_latent, freeze_unused_attn_proj, expected
):
    config = _config(
        likelihood=likelihood,
        sigma_obs=sigma_obs,
        head_structured_latent=head_structured_latent,
        freeze_unused_attn_proj=freeze_unused_attn_proj,
    )

    assert trainer.select_ddp_strategy(8, config) == expected


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


def test_the_override_reaches_the_trainer_kwargs(trainer, monkeypatch):
    """The hook and the builder, joined.

    ``_build_trainer_kwargs`` sets ``strategy`` only under CUDA, so this patch is what makes the
    assertion mean anything on a CPU box. Read under a config only the override sends to the
    fallback: the base's default returns plain ``'ddp'`` for any multi-device run, so a hook the
    framework never looked up (an underscore-prefixed name, say) would pass a ``'ddp'`` check.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer.cuda_devices = [0, 1, 2, 3, 4, 5, 6, 7]
    trainer.config["model_config"]["VAE_model"]["freeze_unused_attn_proj"] = False

    kwargs = trainer._build_trainer_kwargs([])

    assert kwargs["strategy"] == _FUT
    assert kwargs["accelerator"] == "gpu"


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
