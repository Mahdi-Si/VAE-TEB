r"""The permutation control, and the equivalence that lets it stay cheap.

:func:`perm_kl_from_forward` permutes the *already-computed* source state instead of re-encoding
a permuted source stream. That shortcut is legitimate only because the source path has no
batch-coupled operator, so

$$\mathrm{Encoder}\big(\mathrm{Adapter}(\pi(U))\big)_i \;=\; H^u[\pi(i)].$$

If that ever stopped holding -- someone adds a BatchNorm, or any cross-sample statistic, to the
source path -- the fused control would silently start measuring something else. So the
equivalence is asserted directly rather than assumed, and it is the most important test here.

The rest pin what the control's gradient is allowed to touch. ``detach_prior`` matters more than
it looks: without it, the control can be minimised by dragging the *prior* toward $q$ instead of
by collapsing the source-driven deltas -- satisfying the objective while destroying the very
quantity being measured.

**Every KL assertion perturbs the posterior first**, or it would pass on a zero.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.controls import (
    make_derangement,
    perm_forward_outputs,
    perm_kl_from_forward,
    permutation_kl,
)
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

_BATCH = 4
_SEQ_LEN = 16


def _model(prod_kwargs, perturb_posterior, **overrides):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(prod_kwargs, **overrides)).eval()
    perturb_posterior(model)
    return model


def _live_decoder_model(prod_kwargs, perturb_posterior, **overrides):
    """A model whose residual decoder is also off its zero-init.

    The residual decoder's mean head starts zeroed, so ``delta_mu_src`` is identically $0$ no
    matter what latent it is handed. Any assertion that the prediction-space control *changed*
    the forecast is vacuous until this is undone -- the same trap the posterior delta heads set,
    one layer further down.
    """
    model = _model(prod_kwargs, perturb_posterior, **overrides)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        head = model.residual_decoder.mean_head
        head.weight.add_(torch.randn(head.weight.shape, generator=generator) * 0.1)
    return model


def _inputs(batch=_BATCH):
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(batch, _SEQ_LEN, 43, generator=generator),
        torch.randn(batch, _SEQ_LEN, 66, generator=generator),
        torch.randn(batch, _SEQ_LEN, 58, generator=generator),
    )


@pytest.mark.parametrize("head_structured", [False, True])
def test_permuting_the_source_state_equals_re_encoding_a_permuted_source(
    prod_kwargs, perturb_posterior, head_structured
):
    """The equivalence the fused control rests on. Break the source path and this fails first."""
    model = _model(prod_kwargs, perturb_posterior, head_structured_latent=head_structured)
    inputs = _inputs()
    perm = make_derangement(_BATCH, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        h_u = model.source_encoder(model.source_adapter(inputs[2]))
        re_encoded = model.source_encoder(model.source_adapter(inputs[2][perm]))

    assert torch.allclose(h_u[perm], re_encoded, atol=1e-5)


@pytest.mark.parametrize("head_structured", [False, True])
def test_the_fused_control_equals_the_re_encoding_control(
    prod_kwargs, perturb_posterior, head_structured
):
    model = _model(prod_kwargs, perturb_posterior, head_structured_latent=head_structured)
    inputs = _inputs()
    perm = make_derangement(_BATCH, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        fused = perm_kl_from_forward(model, out, perm_index=perm)
        re_encoded = permutation_kl(model, *inputs, perm_index=perm)

    assert torch.allclose(fused["perm_kl"], re_encoded["perm_kl"], atol=1e-5)


def test_the_control_respects_a_supplied_weight(prod_kwargs, perturb_posterior):
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    perm = make_derangement(_BATCH, generator=torch.Generator().manual_seed(0))

    weight = torch.ones(_BATCH, _SEQ_LEN)
    weight[:, _SEQ_LEN // 2 :] = 0.0

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        weighted = perm_kl_from_forward(model, out, perm_index=perm, weight=weight)
        unweighted = perm_kl_from_forward(model, out, perm_index=perm)

    assert not torch.allclose(weighted["perm_kl"], unweighted["perm_kl"])


def _prior_grads(model, detach_prior):
    """Backward the control and report the two prior heads' gradients."""
    inputs = _inputs()
    perm = make_derangement(_BATCH, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    out = model(*inputs)
    perm_kl_from_forward(model, out, perm_index=perm, detach_prior=detach_prior)[
        "perm_kl"
    ].backward()
    return (
        model.prior_head.mu_prior_head.body[0].weight.grad,
        model.prior_head.logvar_prior_head.body[0].weight.grad,
    )


def test_the_prior_mean_head_is_structurally_unreachable_from_the_kl(
    prod_kwargs, perturb_posterior
):
    r"""$\mu^q = \mu^p + \Delta\mu$, so $\mu^q - \mu^p$ cancels $\mu^p$ exactly.

    The KL's mean term therefore has *no* dependence on the prior mean, whatever
    ``detach_prior`` says. Worth pinning, because it means a ``detach_prior`` test written
    against this head would pass for the wrong reason and prove nothing -- the log-variance
    head below is where the flag actually does its work.
    """
    mu_grad, _ = _prior_grads(_model(prod_kwargs, perturb_posterior), detach_prior=False)
    assert mu_grad is None or mu_grad.abs().max().item() == 0.0


def test_detach_prior_routes_gradient_away_from_the_prior(prod_kwargs, perturb_posterior):
    """Without this the control could be satisfied by moving the prior, not the deltas."""
    _, logvar_grad = _prior_grads(_model(prod_kwargs, perturb_posterior), detach_prior=True)
    assert logvar_grad is None or logvar_grad.abs().max().item() == 0.0


def test_without_detach_prior_the_prior_does_receive_gradient(prod_kwargs, perturb_posterior):
    """The mirror image: the test above must be capable of failing."""
    _, logvar_grad = _prior_grads(_model(prod_kwargs, perturb_posterior), detach_prior=False)
    assert logvar_grad is not None and logvar_grad.abs().max().item() > 0.0


def test_the_control_carries_gradient_into_the_posterior(prod_kwargs, perturb_posterior):
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    perm = make_derangement(_BATCH, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(0)
    out = model(*inputs)
    perm_kl_from_forward(model, out, perm_index=perm)["perm_kl"].backward()

    grad = model.posterior_head.delta_mu_head.weight.grad
    assert grad is not None and grad.abs().max().item() > 0.0


def test_the_readout_is_detached(prod_kwargs, perturb_posterior):
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    out = model(*inputs)
    result = perm_kl_from_forward(model, out)

    assert result["perm_kl"].requires_grad
    assert not result["kld_shuffled"].requires_grad
    assert not result["kld_shuffled_per_t"].requires_grad


def test_the_shuffled_curve_matches_the_true_curves_semantics(prod_kwargs, perturb_posterior):
    """Both are raw, full-T and summed over latent dims, so they can be plotted together."""
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        result = perm_kl_from_forward(model, out)

    assert result["kld_shuffled_per_t"].shape == out["kld_per_t"].shape


def test_the_control_uses_a_real_derangement(prod_kwargs, perturb_posterior):
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        result = perm_kl_from_forward(model, out)

    perm = result["perm_index"]
    assert not bool((perm == torch.arange(_BATCH)).any())


def test_a_degenerate_batch_cannot_be_deranged(prod_kwargs, perturb_posterior):
    """B < 2 has no derangement; the caller must skip the control rather than fake one."""
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs(batch=1)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    with pytest.raises(ValueError, match="batch_size >= 2"):
        perm_kl_from_forward(model, out)


def test_a_wrong_shaped_perm_index_raises(prod_kwargs, perturb_posterior):
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    with pytest.raises(ValueError, match="perm_index must have shape"):
        perm_kl_from_forward(model, out, perm_index=torch.arange(3))


def test_the_prediction_space_control_rebuilds_only_the_source_driven_tensors(
    prod_kwargs, perturb_posterior
):
    """The encoders, the prior and the target-only baseline must be reused untouched."""
    model = _live_decoder_model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        permuted = perm_forward_outputs(model, out, perm_index=make_derangement(_BATCH))

    for shared in ("target_state", "source_state", "mu_prior", "mu_base", "decoder_state"):
        assert torch.equal(permuted[shared], out[shared]), f"{shared} was rebuilt"
    for rebuilt in ("mu_post", "z", "delta_mu_src", "mu_full", "logvar_full"):
        assert not torch.equal(permuted[rebuilt], out[rebuilt]), f"{rebuilt} was not rebuilt"


def test_the_permuted_forecast_feeds_compute_loss(prod_kwargs, perturb_posterior):
    """The control's whole purpose: re-score the forecast under a stranger's source."""
    model = _model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        permuted = perm_forward_outputs(model, out, perm_index=make_derangement(_BATCH))
        shuffled = model.compute_loss(
            permuted, inputs[0], inputs[1], compute_kld_loss=False
        )
        true = model.compute_loss(out, inputs[0], inputs[1], compute_kld_loss=False)

    assert torch.isfinite(shuffled["feat_loss"])
    # The baseline is target-only, so it is untouched by permuting the source.
    assert torch.allclose(shuffled["base_loss"], true["base_loss"], atol=1e-6)


def test_the_permuted_rebuild_is_full_mu_base_plus_the_new_delta(prod_kwargs, perturb_posterior):
    model = _live_decoder_model(prod_kwargs, perturb_posterior)
    inputs = _inputs()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
        permuted = perm_forward_outputs(model, out, perm_index=make_derangement(_BATCH))

    assert torch.allclose(
        permuted["mu_full"], out["mu_base"] + permuted["delta_mu_src"], atol=1e-6
    )
