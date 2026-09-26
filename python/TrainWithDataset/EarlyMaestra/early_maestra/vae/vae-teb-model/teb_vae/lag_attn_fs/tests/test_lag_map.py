r"""The lag-resolved KL attribution, on this model: an exact decomposition, not an approximation.

The attribution is a statement about the latent and the attention and says nothing about what is
being forecast, so it is the sibling's code reached by inheritance, and its per-head split and
non-negativity are pinned in the sibling's suite. The conservation identity is re-asserted here, on
this class at a gated target width: the decomposition is only rigorous while the posterior stays
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
identically zero, and the identity holds vacuously on any model at all.
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
def test_the_identity_holds_at_a_gated_target_width(
    tiny_gated, inputs, perturb_posterior, use_entmax
):
    """The one thing that is genuinely this model's: the decoder's width follows the reach
    budget's surviving target channels. The attribution must not notice -- it is computed from the
    latent and the attention, neither of which the target gate touches -- and a decomposition that
    had picked up a dependence on the output width would show here and nowhere else. Both
    attention normalisers, because ``entmax`` is what the shipped config runs and it is the one
    whose weights can be exactly zero on some lags."""
    out = _perturbed_forward(tiny_gated, inputs, perturb_posterior, use_entmax=use_entmax)

    assert torch.allclose(
        out["source_kl_lag_map"].sum(dim=-1), out["kld_per_t"], atol=_ROUNDOFF, rtol=_ROUNDOFF
    )
