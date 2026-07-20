"""The shared fixtures are non-vacuous: the checkpoint is self-describing, the perturbations bite.

The second half is the point of the file. Every later test that asserts on a KL, an uplift, a
residual ratio or a lag-band difference rests on one of the two perturbation fixtures, and
which one it needs is not obvious: ``_zero_init_delta_heads`` zeroes the residual decoder's
mean head as well as the posterior deltas, so a ``perturb_posterior`` model has
``delta_mu_src`` identically zero *whatever* $z$ is. Tests of the forecast pathway written
against it pass while proving nothing. These assertions pin that distinction so it cannot
quietly stop holding.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.eval.tests.conftest import build_tiny_checkpoint_blob
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn


def _delta_mu_src(model, inputs) -> torch.Tensor:
    """Run a forward and return the source-driven mean shift."""
    model.eval()
    with torch.no_grad():
        return model(*inputs)["delta_mu_src"]


# ---------------------------------------------------------------------------
# The checkpoint fixture
# ---------------------------------------------------------------------------
def test_tiny_checkpoint_is_self_describing(tiny_checkpoint):
    """Both stamps present: the base's ``model_class`` and the task's ``model_kwargs``."""
    blob = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeLagAttn"
    # Asserted explicitly because the failure is silent: a task built without ``model_kwargs=``
    # stores {}, and ``SeqVaeLagAttn(**{})`` then builds the full production geometry rather
    # than raising -- a 300-step, 128-wide model reporting itself as the tiny one.
    assert blob["model_kwargs"], "empty model_kwargs would silently rebuild production geometry"
    assert blob["model_kwargs"]["d_model"] == 32
    assert blob["hyper_parameters"]["likelihood"] == "gaussian_nll"


def test_tiny_checkpoint_round_trips(tiny_checkpoint):
    """Rebuilt from its own ``model_kwargs``, the state dict aligns parameter for parameter."""
    blob = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    model = SeqVaeLagAttn(**blob["model_kwargs"])

    # The task stores the net under both ``model`` and ``_orig_model``, so strip either prefix.
    state = {}
    for key, value in blob["state_dict"].items():
        for prefix in ("model.", "_orig_model."):
            if key.startswith(prefix):
                state[key[len(prefix):]] = value
                break
    # strict=True raises on any disagreement, so reaching the assertion is most of the check.
    result = model.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []


def test_tiny_checkpoint_weights_differ_from_a_fresh_build(tiny_checkpoint):
    """The fixture must not be indistinguishable from a checkpoint that never loaded.

    A freshly constructed model has zero delta heads. So does a model whose checkpoint silently
    failed to load. A fixture built without perturbation would therefore fail the very load
    verification it exists to demonstrate passing.
    """
    blob = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    saved = blob["state_dict"]["model.residual_decoder.mean_head.weight"]
    assert not torch.allclose(saved, torch.zeros_like(saved))


def test_unperturbed_checkpoint_option_is_zero_init():
    """The escape hatch used by the preflight tests: a blob that *should* fail verification."""
    blob = build_tiny_checkpoint_blob(perturb=False)
    saved = blob["state_dict"]["model.residual_decoder.mean_head.weight"]
    assert torch.allclose(saved, torch.zeros_like(saved))


# ---------------------------------------------------------------------------
# The perturbation fixtures
# ---------------------------------------------------------------------------
def test_perturb_posterior_leaves_delta_mu_src_identically_zero(
    shipped_kwargs, inputs, perturb_posterior
):
    """The trap, asserted directly.

    Perturbing the posterior makes the KL nonzero, which is what that fixture is for -- but the
    residual decoder's mean head is still at its zero init, so the source-driven mean shift is
    exactly zero regardless of $z$, and every forecast-pathway assertion on this model is
    vacuous.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**shipped_kwargs)
    perturb_posterior(model)

    delta = _delta_mu_src(model, inputs)
    assert torch.count_nonzero(delta) == 0

    # ... while the KL genuinely moved, which is the whole reason the fixture looks sufficient.
    with torch.no_grad():
        outputs = model(*inputs)
        kld = model.kld_tensor(
            mu_prior=outputs["mu_prior"],
            logvar_prior=outputs["logvar_prior"],
            mu_post=outputs["mu_post"],
            logvar_post=outputs["logvar_post"],
        )
    assert float(kld.abs().sum()) > 0.0


def test_perturb_full_pathway_opens_the_forecast_residual(
    shipped_kwargs, inputs, perturb_full_pathway
):
    """The fixture every uplift / residual / ablation test requires."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**shipped_kwargs)
    perturb_full_pathway(model)

    delta = _delta_mu_src(model, inputs)
    assert torch.count_nonzero(delta) > 0
    assert torch.isfinite(delta).all()


def test_perturbations_are_deterministic(shipped_kwargs, inputs, perturb_full_pathway):
    """A fixed seed must give the same model twice, or no numeric assertion downstream holds."""
    deltas = []
    for _ in range(2):
        torch.manual_seed(0)
        model = SeqVaeLagAttn(**shipped_kwargs)
        perturb_full_pathway(model)
        torch.manual_seed(11)  # forward samples z; pin it too
        deltas.append(_delta_mu_src(model, inputs))
    assert torch.equal(deltas[0], deltas[1])
