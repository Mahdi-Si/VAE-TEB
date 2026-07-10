r"""S4-T04: v3 resolves to plain ``'ddp'``, and earns the right to.

``find_unused_parameters=False`` (what plain ``'ddp'`` implies) is only safe when the DDP
reducer's expectation holds: *every* parameter is marked ready in *every* backward. v1 gates
that on the decoder log-variance heads, which receive gradients only under
``likelihood='gaussian_nll'`` **and** ``sigma_obs='learned'`` -- exactly v3's defaults (G7).

The strategy string is a claim; the grad-coverage test below is the evidence. It is run under
both the ``head_structured_latent`` variants and on both step types, because the fused
permutation control (G6) changes the shape of the autograd graph on scheduled steps.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.trainer_lag_attn_v3 import (
    GraphModelVaeTebLagAttnV3Trainer,
)

_select = GraphModelVaeTebLagAttnV3Trainer._select_ddp_strategy


def test_learned_variance_selects_plain_ddp():
    assert _select(4, "gaussian_nll", "learned") == "ddp"
    assert _select(8, "gaussian_nll", "learned") == "ddp"


def test_production_config_as_written_resolves_to_plain_ddp():
    """S7-T01: the shipped config (``sigma_obs: learned`` + frozen ``W_o``) earns plain 'ddp'.

    The trainer defaults ``sigma_obs`` to ``'learned'`` but the YAML value wins, so a config
    left at the fixed-variance debug scalar ``1.0`` would silently pick the slow strategy and
    leave the learned-variance head (and the calibration report) untrained. This binds the
    config on disk to the resolved strategy.
    """
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "config_lag_attn_v3.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        vae = yaml.safe_load(fh)["model_config"]["VAE_model"]

    assert vae["likelihood"] == "gaussian_nll"
    assert vae["sigma_obs"] == "learned", "production config must use learned observation variance"
    strategy = _select(
        8, vae["likelihood"], vae["sigma_obs"],
        head_structured_latent=bool(vae.get("head_structured_latent", False)),
        freeze_unused_attn_proj=bool(vae.get("freeze_unused_attn_proj", False)),
    )
    assert strategy == "ddp", strategy


@pytest.mark.parametrize(
    "likelihood,sigma_obs",
    [("mse", 1.0), ("mse", "learned"), ("gaussian_nll", 1.0), ("gaussian_nll", 0.5)],
)
def test_unconsumed_logvar_heads_require_find_unused(likelihood, sigma_obs):
    """Any config that leaves the decoder logvar heads dangling needs the slow strategy."""
    assert _select(4, likelihood, sigma_obs) == "ddp_find_unused_parameters_true"


def test_head_structured_latent_starves_the_attention_projection():
    """v3-only: ``lag_attn.W_o`` is dead in head-structured mode unless it is frozen.

    v1 never hit this because its fixed ``sigma_obs`` already forced the slow strategy. v3's
    learned variance removes that cover, so the production config -- which *does* set
    ``head_structured_latent: true`` -- would otherwise pick plain ``'ddp'`` and crash the run.
    """
    assert _select(
        4, "gaussian_nll", "learned", head_structured_latent=True
    ) == "ddp_find_unused_parameters_true"
    assert _select(
        4, "gaussian_nll", "learned",
        head_structured_latent=True, freeze_unused_attn_proj=True,
    ) == "ddp"


def test_single_device_needs_no_strategy():
    assert _select(1, "gaussian_nll", "learned") == "auto"
    assert _select(0, "mse", 1.0) == "auto"


@pytest.mark.parametrize("head_structured", [False, True])
@pytest.mark.parametrize("batch_idx", [0, 1])  # 0 = permutation step, 1 = plain step
def test_no_parameter_is_left_without_a_gradient(
    v3_pl, stub_batch, perturb_posterior, prod_kwargs, head_structured, batch_idx
):
    """The invariant that actually licenses ``find_unused_parameters=False``."""
    pl_module = v3_pl(model_kwargs=dict(prod_kwargs, head_structured_latent=head_structured))
    perturb_posterior(pl_module.orig_model)

    pl_module.zero_grad(set_to_none=True)
    loss, _ = pl_module.compute_loss_and_metrics(stub_batch, batch_idx, "train")
    loss.backward()

    starved = [
        name
        for name, param in pl_module.orig_model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not starved, (
        f"head_structured={head_structured}, batch_idx={batch_idx}: these parameters got no "
        f"gradient, so plain 'ddp' would raise or deadlock: {starved}"
    )


def test_freezing_is_what_removes_w_o_from_the_expectation_set(
    v3_pl, stub_batch, perturb_posterior, prod_kwargs
):
    """Without the freeze, ``W_o`` is trainable and starved -- exactly the crash condition."""
    unfrozen = v3_pl(
        model_kwargs=dict(
            prod_kwargs, head_structured_latent=True, freeze_unused_attn_proj=False
        )
    )
    perturb_posterior(unfrozen.orig_model)
    assert unfrozen.orig_model.frozen_attn_proj is False

    unfrozen.zero_grad(set_to_none=True)
    loss, _ = unfrozen.compute_loss_and_metrics(stub_batch, 1, "train")
    loss.backward()

    starved = [
        name
        for name, param in unfrozen.orig_model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert starved == ["lag_attn.W_o.weight", "lag_attn.W_o.bias"], starved

    frozen = v3_pl(model_kwargs=dict(prod_kwargs, head_structured_latent=True))
    assert frozen.orig_model.frozen_attn_proj is True
    assert not frozen.orig_model.lag_attn.W_o.weight.requires_grad


def test_freezing_is_a_no_op_in_the_flat_posterior(v3_pl, prod_kwargs):
    """In flat mode ``W_o`` feeds the posterior, so it must stay trainable regardless."""
    flat = v3_pl(model_kwargs=dict(prod_kwargs, head_structured_latent=False))
    assert flat.orig_model.frozen_attn_proj is False
    assert flat.orig_model.lag_attn.W_o.weight.requires_grad


def test_freezing_does_not_change_the_forward(prod_kwargs, inputs):
    """The freeze must be numerically invisible -- it only touches ``requires_grad``."""
    from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3

    kwargs = dict(prod_kwargs, head_structured_latent=True)
    torch.manual_seed(0)
    frozen = SeqVaeLagAttnV3(**dict(kwargs, freeze_unused_attn_proj=True)).eval()
    torch.manual_seed(0)
    trainable = SeqVaeLagAttnV3(**dict(kwargs, freeze_unused_attn_proj=False)).eval()

    torch.manual_seed(5)
    a = frozen(*inputs)
    torch.manual_seed(5)
    b = trainable(*inputs)
    for key in ("attended_source", "mu_post", "mu_full", "kld_per_t"):
        assert torch.allclose(a[key], b[key], atol=1e-6), f"freeze changed {key}"


def test_decoder_logvar_heads_are_the_ones_that_matter(v3_pl, stub_batch, prod_kwargs):
    """Under the fixed-sigma config the logvar heads really are starved (v1's rule holds)."""
    pl_module = v3_pl(hparams={"likelihood": "mse", "sigma_obs": 1.0})
    pl_module.zero_grad(set_to_none=True)
    loss, _ = pl_module.compute_loss_and_metrics(stub_batch, 1, "train")
    loss.backward()

    logvar_head_params = [
        name
        for name, param in pl_module.orig_model.named_parameters()
        if ".logvar_head." in name and param.grad is None
    ]
    assert logvar_head_params, (
        "the decoder logvar heads received gradients under likelihood='mse'; "
        "_select_ddp_strategy's rule no longer describes the model"
    )


def test_learned_variance_actually_reaches_the_logvar_heads(v3_pl, stub_batch):
    pl_module = v3_pl()  # PROD_HPARAMS: gaussian_nll + learned
    pl_module.zero_grad(set_to_none=True)
    loss, metrics = pl_module.compute_loss_and_metrics(stub_batch, 1, "train")
    loss.backward()

    grads = [
        param.grad
        for name, param in pl_module.orig_model.named_parameters()
        if ".logvar_head." in name and param.grad is not None
    ]
    assert grads, "sigma_obs='learned' did not reach the decoder logvar heads"
    assert any(g.abs().max() > 0 for g in grads)
    assert torch.isfinite(metrics["mean_logvar_full"])
