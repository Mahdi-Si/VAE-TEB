r"""The lag-resolved KL attribution: an exact decomposition, not an approximation.

Because the posterior is head-structured, latent group $m$ is written only by lag-attention head $m$,
and

$$\mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} \alpha^{(m)}_{t,\ell}
\quad\Rightarrow\quad \sum_\ell \mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} = K_t,$$

since each head's weights sum to one over its valid lags. Attention dropout would break that identity
pointwise -- it would hold only in expectation -- and the lag attention is built at zero dropout
precisely so it holds per element. What remains is float summation over $L$ lags, so the tolerances
below are round-off tolerances rather than statistical ones, and they are named.

Kept in this package despite the delegation test proving ``compute_loss`` is the sibling's, because
the lag map is not produced by ``compute_loss`` at all: it comes out of the ``te_analysis`` call
inside ``forward``, from tensors this model's own forward assembled. A lag axis that arrived
transposed, or an attention fed the wrong state, would leave the two ``compute_loss`` calls in
perfect agreement and this identity broken.

**The row-sum identity alone cannot see a mis-wired** ``head_structured`` **flag**, and it is worth
saying why rather than leaving the impression that it can. The flag selects only the map's form:
$\sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$ when set, and $K_t \bar\alpha_{t,\ell}$ against the
head-mean attention when clear. Every head's weights sum to one over lags, so the head mean does
too, and *both* forms sum over $\ell$ to exactly $K_t$ -- as does the per-head split, which
``te_analysis`` computes identically either way. So the flag is pinned separately below, against the
two forms recomputed from the model's own returned tensors.

There is one thing this readout means here that it does not mean for the sibling, and it is the
package's whole point. There, the input features at step $t$ read hundreds of seconds into their own
future, so the quantity is a source-conditioned KL and must not be named for the transfer entropy it
is not. Here the history is strictly one-sided by construction -- but the name stays
``source_kl_lag_map`` regardless, because what the model reports is still a KL between two of its own
distributions, and renaming a quantity on the strength of an architectural argument is how a readout
starts being read as something it was never measured to be.

Every test perturbs the posterior first. At initialisation the KL is identically zero, the map is
identically zero, and all three identities hold vacuously on any model at all.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E

#: Agreement required between the lag map's row sum and the per-step KL, and between the per-head
#: split and the same total. **Relative**, because the identity is exact and what is left is float32
#: round-off on a sum whose terms are $O(K_t)$: on a perturbed model $K_t$ reaches $10^2$ nats, so an
#: absolute bound would be a statement about the perturbation's scale rather than about the
#: decomposition.
_SUM_RTOL = 1e-6

#: Absolute floor beside it, for anchors whose KL is genuinely near zero and where a relative bound
#: is meaningless.
_SUM_ATOL = 1e-6

#: How far below zero a non-negative quantity may fall before it is a wrong tensor rather than a
#: cancellation in the last bits.
_NONNEG_TOL = -1e-7


def _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, **overrides):
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**dict(tiny_kwargs, **overrides)).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    assert float(out["kld_per_t"].abs().max()) > 0.0, "perturbation failed; test is vacuous"
    return out


@pytest.mark.parametrize("use_entmax", [False, True])
def test_the_lag_map_sums_over_lags_to_the_per_step_kl(
    tiny_kwargs, raw_inputs, perturb_posterior, use_entmax
):
    """Both attention families, because ``entmax15`` can assign a lag exactly zero weight and a
    decomposition that quietly renormalised would still sum to one under softmax."""
    out = _perturbed_forward(tiny_kwargs, raw_inputs, perturb_posterior, use_entmax=use_entmax)
    total = out["source_kl_lag_map"].sum(dim=-1)

    assert torch.allclose(total, out["kld_per_t"], atol=_SUM_ATOL, rtol=_SUM_RTOL)


def test_the_per_head_decomposition_sums_to_the_total(tiny_kwargs, raw_inputs, perturb_posterior):
    out = _perturbed_forward(tiny_kwargs, raw_inputs, perturb_posterior)

    assert torch.allclose(
        out["kld_per_t_per_head"].sum(dim=-1), out["kld_per_t"], atol=_SUM_ATOL, rtol=_SUM_RTOL
    )


def test_the_map_is_nonnegative(tiny_kwargs, raw_inputs, perturb_posterior):
    """Per-group KLs are sums of per-dimension KLs (each $\\ge 0$) and the weights are probabilities,
    so a negative cell means the decomposition read the wrong tensors."""
    out = _perturbed_forward(tiny_kwargs, raw_inputs, perturb_posterior)

    assert float(out["source_kl_lag_map"].min()) >= _NONNEG_TOL
    assert float(out["kld_per_t_per_head"].min()) >= _NONNEG_TOL


def test_the_identity_survives_an_encoder_head_count_of_its_own(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    """The encoder's self-attention heads are unrelated to the lag-attention heads and to the latent
    groups. They happen to number the same at the shipped configuration, so an accidental coupling
    would go unnoticed until a depth-or-width arm changed one of them."""
    out = _perturbed_forward(tiny_kwargs, raw_inputs, perturb_posterior, encoder_num_heads=2)
    total = out["source_kl_lag_map"].sum(dim=-1)

    assert out["kld_per_t_per_head"].shape[-1] == int(tiny_kwargs["num_heads"])
    assert torch.allclose(total, out["kld_per_t"], atol=_SUM_ATOL, rtol=_SUM_RTOL)


def test_the_map_is_the_head_structured_form_and_not_the_head_mean_one(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    r"""The assertion that actually pins ``head_structured=True`` at the ``te_analysis`` call.

    Both forms of the map sum over lags to $K_t$, so the identity above passes either way -- this
    file's own docstring records that. What separates them is the map itself, per lag:

    $$\mathrm{structured}_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell},
      \qquad \mathrm{mean}_{t,\ell} = K_t \cdot \tfrac1M \sum_m \alpha^{(m)}_{t,\ell}.$$

    Both are recomputed here from the model's **own returned tensors**, so the comparison is against
    the arithmetic rather than against a second call into the head. The model must match the first
    and differ from the second; the second half is the negative control, and it is not free -- the
    two coincide exactly when every head carries the same KL, which is why the posterior is
    perturbed first.
    """
    out = _perturbed_forward(tiny_kwargs, raw_inputs, perturb_posterior)
    per_head, alpha = out["kld_per_t_per_head"], out["attn_weights"]

    structured = torch.einsum("btm,btml->btl", per_head, alpha)
    head_mean = out["kld_per_t"].unsqueeze(-1) * alpha.mean(dim=-2)

    assert torch.allclose(
        out["source_kl_lag_map"], structured, atol=_SUM_ATOL, rtol=_SUM_RTOL
    ), "the lag map is not the head-structured attribution; head_structured is not set"
    # The control: the two forms must be genuinely distinguishable on this batch, or the assertion
    # above would hold against a model built with the flag clear.
    assert not torch.allclose(structured, head_mean, atol=_SUM_ATOL, rtol=_SUM_RTOL), (
        "the two attribution forms agree on this batch, so the assertion above proves nothing"
    )


def test_the_readout_is_named_for_the_kl_not_transfer_entropy(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    """The history is one-sided here, which removes the acausality objection -- but a KL between two
    of the model's own distributions is what was measured, and that is what the key says. Renaming it
    would make an architectural argument look like a measurement."""
    out = _perturbed_forward(tiny_kwargs, raw_inputs, perturb_posterior)

    assert "source_kl_lag_map" in out
    assert not any("te_lag" in key for key in out)
