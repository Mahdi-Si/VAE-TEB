r"""The lag-resolved KL attribution, on this model: an exact decomposition, not an approximation.

The attribution is a statement about the latent and the attention and says nothing about what is
being forecast, so it is the sibling's code reached by inheritance. That is exactly why it is
re-asserted here rather than assumed: the decomposition is only rigorous while the posterior stays
head-structured, and a target-domain subclass that quietly changed the posterior or the head
grouping would leave the sibling's suite green and this model's central readout meaningless.

Because the posterior is head-structured, latent group $m$ is written only by attention head $m$,
and

$$\mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} \alpha^{(m)}_{t,\ell}
\quad\Rightarrow\quad \sum_\ell \mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} = K_t,$$

since each head's weights sum to one over its valid lags.

**The identity is exact in arithmetic and not in float32.** Attention dropout would break it
pointwise -- it would then hold only in expectation -- and the model is built with it at zero, so
what is demanded below is round-off agreement rather than a statistical tolerance. It is *not*
``torch.equal``: the map is contracted with ``einsum`` over the head axis while $K_t$ is summed
over the dimension axis, so the two reach the same number by different summation orders and differ
in the last bits. Measured at these geometries the gap is $\approx 10^{-6}$ on values of order
$10$, against the $10^{-5}$ asserted here.

Every test perturbs the posterior first. At initialisation the KL is identically zero, the map is
identically zero, and all three identities hold vacuously on any model at all.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs

#: Agreement demanded of the conservation identities. See the module docstring: exact equality is
#: unreachable because the two sides sum over different axes in different orders.
_ROUNDOFF = 1e-5


def _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, **overrides):
    """Build this model, break its zero-init posterior, and run one forward.

    Args:
        tiny_kwargs: The tiny constructor kwargs.
        inputs: The seeded ``(y_st, y_ph, u_stream)`` triple.
        perturb_posterior: The perturbation factory fixture.
        **overrides: Constructor overrides.

    Returns:
        The forward dict.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(tiny_kwargs, **overrides)).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    assert float(out["kld_per_t"].abs().max()) > 0.0, "perturbation failed; test is vacuous"
    return out


@pytest.mark.parametrize("use_entmax", [False, True])
def test_the_lag_map_sums_over_lags_to_the_per_step_kl(
    tiny_kwargs, inputs, perturb_posterior, use_entmax
):
    """Both attention normalisers, because ``entmax`` is what the shipped config runs and it is
    the one whose weights can be exactly zero on some lags."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, use_entmax=use_entmax)

    total = out["source_kl_lag_map"].sum(dim=-1)

    assert torch.allclose(total, out["kld_per_t"], atol=_ROUNDOFF, rtol=_ROUNDOFF)


def test_the_per_head_decomposition_sums_to_the_total(tiny_kwargs, inputs, perturb_posterior):
    """The other half of the identity: the map is the per-head KL contracted with the per-head
    attention, so a per-head split that did not itself add back to $K_t$ would make the lag map
    wrong in a way the test above cannot see."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)

    assert torch.allclose(
        out["kld_per_t_per_head"].sum(dim=-1), out["kld_per_t"], atol=_ROUNDOFF
    )


def test_the_map_is_nonnegative(tiny_kwargs, inputs, perturb_posterior):
    """Per-group KLs are sums of per-dimension KLs (each $\\ge 0$) and the weights are
    probabilities, so a negative cell means the decomposition read the wrong tensors."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)

    assert float(out["source_kl_lag_map"].min()) >= -1e-7
    assert float(out["kld_per_t_per_head"].min()) >= -1e-7


def test_the_readout_is_named_for_the_kl_not_transfer_entropy(
    tiny_kwargs, inputs, perturb_posterior
):
    """The quantity is a source-conditioned KL. Naming it transfer entropy would claim an
    identity the model does not establish, and the name is what a reader carries away."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)

    assert "source_kl_lag_map" in out
    assert "te_lag_map" not in out
    assert not any("te_lag" in key for key in out)


def test_the_identity_holds_at_a_gated_target_width(
    tiny_gated, inputs, perturb_posterior
):
    """The one thing that is genuinely this model's: the decoder's width follows the reach
    budget's surviving target channels. The attribution must not notice -- it is computed from the
    latent and the attention, neither of which the target gate touches -- and a decomposition that
    had picked up a dependence on the output width would show here and nowhere else."""
    out = _perturbed_forward(tiny_gated, inputs, perturb_posterior)

    assert torch.allclose(
        out["source_kl_lag_map"].sum(dim=-1), out["kld_per_t"], atol=_ROUNDOFF, rtol=_ROUNDOFF
    )
