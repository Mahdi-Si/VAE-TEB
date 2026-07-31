r"""The lag-resolved KL attribution: an exact decomposition, not an approximation.

Because the posterior is head-structured, latent group $m$ is written only by lag-attention head
$m$, and

$$\mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} \alpha^{(m)}_{t,\ell}
\quad\Rightarrow\quad \sum_\ell \mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} = K_t,$$

since each head's weights sum to one over its valid lags. Attention dropout would break that
identity pointwise -- it would hold only in expectation -- and the lag attention is built at zero
dropout precisely so it holds per element. What remains is float summation over $L$ lags, so the
tolerances below are round-off tolerances rather than statistical ones, and they are named.

Every test perturbs the posterior first. At initialisation the KL is identically zero, the map is
identically zero, and all three identities hold vacuously on any model at all.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: Agreement required between the lag map's row sum and the per-step KL, and between the per-head
#: split and the same total. **Relative**, because the identity is exact and what is left is
#: float32 round-off on a sum whose terms are $O(K_t)$: on a perturbed model $K_t$ reaches $10^2$
#: nats, so an absolute bound would be a statement about the perturbation's scale rather than about
#: the decomposition. Measured worst case is $2 \times 10^{-7}$, which is float32 eps; $10^{-6}$
#: is a few multiples of it and would not survive a decomposition that read the wrong tensors.
_SUM_RTOL = 1e-6

#: Absolute floor beside it, for anchors whose KL is genuinely near zero and where a relative
#: bound is meaningless.
_SUM_ATOL = 1e-6

#: How far below zero a non-negative quantity may fall before it is a wrong tensor rather than a
#: cancellation in the last bits.
_NONNEG_TOL = -1e-7


def _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, **overrides):
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**dict(tiny_kwargs, **overrides)).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    assert float(out["kld_per_t"].abs().max()) > 0.0, "perturbation failed; test is vacuous"
    return out


@pytest.mark.parametrize("use_entmax", [False, True])
def test_the_lag_map_sums_over_lags_to_the_per_step_kl(
    tiny_kwargs, inputs, perturb_posterior, use_entmax
):
    """Both attention families, because ``entmax15`` can assign a lag exactly zero weight and a
    decomposition that quietly renormalised would still sum to one under softmax."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, use_entmax=use_entmax)
    total = out["source_kl_lag_map"].sum(dim=-1)

    assert torch.allclose(total, out["kld_per_t"], atol=_SUM_ATOL, rtol=_SUM_RTOL)


def test_the_per_head_decomposition_sums_to_the_total(tiny_kwargs, inputs, perturb_posterior):
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)

    assert torch.allclose(
        out["kld_per_t_per_head"].sum(dim=-1), out["kld_per_t"], atol=_SUM_ATOL, rtol=_SUM_RTOL
    )


def test_the_map_is_nonnegative(tiny_kwargs, inputs, perturb_posterior):
    """Per-group KLs are sums of per-dimension KLs (each $\\ge 0$) and the weights are
    probabilities, so a negative cell means the decomposition read the wrong tensors."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)

    assert float(out["source_kl_lag_map"].min()) >= _NONNEG_TOL
    assert float(out["kld_per_t_per_head"].min()) >= _NONNEG_TOL


def test_the_identity_survives_an_encoder_head_count_of_its_own(
    tiny_kwargs, inputs, perturb_posterior
):
    """The encoder's self-attention heads are unrelated to the lag-attention heads and to the
    latent groups. They happen to number the same at the shipped configuration, so an accidental
    coupling would go unnoticed until a depth-or-width arm changed one of them."""
    out = _perturbed_forward(
        tiny_kwargs, inputs, perturb_posterior, encoder_num_heads=2
    )
    total = out["source_kl_lag_map"].sum(dim=-1)

    assert out["kld_per_t_per_head"].shape[-1] == int(tiny_kwargs["num_heads"])
    assert torch.allclose(total, out["kld_per_t"], atol=_SUM_ATOL, rtol=_SUM_RTOL)


def test_the_readout_is_named_for_the_kl_not_transfer_entropy(
    tiny_kwargs, inputs, perturb_posterior
):
    """Under ``causal_reach_budget_s: null`` the input features at step $t$ read far into their own
    future, so this quantity is a source-conditioned KL and must not be named for the transfer
    entropy it is not yet."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)

    assert "source_kl_lag_map" in out
    assert not any("te_lag" in key for key in out)
