r"""The evidence behind the DDP strategy the configured likelihood selects, on this model.

``find_unused_parameters=False`` makes the reducer expect every parameter to be marked ready in
every backward, and one that is not raises or deadlocks on a real multi-GPU box. The selector is the
sibling's and is tested there; what this file measures is whether its claim survives the
target-domain change. The one config-decided starvation source is the decoder log-variance heads,
consumed only under ``likelihood: gaussian_nll`` -- and those are exactly the tensors this model
widened from $R$ outputs to $C_{\mathrm{keep}}$, so if the change were going to move the starved set
this is where it would show.
"""
from __future__ import annotations

import pytest

from .conftest import make_stub_batch


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
    anchor weights, because the prior scale rate is the one term a config can switch on.
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
