r"""The lag-resolved KL attribution: an exact decomposition, not an approximation.

Because the posterior is head-structured, latent group $m$ is written only by attention head
$m$, and

$$\mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} \alpha^{(m)}_{t,\ell}
\quad\Rightarrow\quad \sum_\ell \mathrm{map}_{t,\ell} = \sum_m K_t^{(m)} = K_t,$$

since each head's weights sum to one over its valid lags. Attention dropout would break that
identity pointwise (it would hold only in expectation); the model is built with it at zero, so
the tests below demand float-round-off agreement, not a statistical tolerance.

Every test perturbs the posterior first. At initialisation the KL is identically zero, the map
is identically zero, and all three identities hold vacuously on any model at all.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, **overrides):
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(tiny_kwargs, **overrides)).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    assert float(out["kld_per_t"].abs().max()) > 0.0, "perturbation failed; test is vacuous"
    return out


@pytest.mark.parametrize("use_entmax", [False, True])
def test_the_lag_map_sums_over_lags_to_the_per_step_kl(
    tiny_kwargs, inputs, perturb_posterior, use_entmax
):
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior, use_entmax=use_entmax)
    total = out["source_kl_lag_map"].sum(dim=-1)
    assert torch.allclose(total, out["kld_per_t"], atol=1e-5, rtol=1e-5)


def test_the_per_head_decomposition_sums_to_the_total(
    tiny_kwargs, inputs, perturb_posterior
):
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)
    assert torch.allclose(
        out["kld_per_t_per_head"].sum(dim=-1), out["kld_per_t"], atol=1e-6
    )


def test_the_map_is_nonnegative(tiny_kwargs, inputs, perturb_posterior):
    """Per-group KLs are sums of per-dimension KLs (each >= 0) and the weights are
    probabilities, so a negative cell means the decomposition read the wrong tensors."""
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)
    assert float(out["source_kl_lag_map"].min()) >= -1e-7
    assert float(out["kld_per_t_per_head"].min()) >= -1e-7


def test_the_readout_is_named_for_the_kl_not_transfer_entropy(
    tiny_kwargs, inputs, perturb_posterior
):
    out = _perturbed_forward(tiny_kwargs, inputs, perturb_posterior)
    assert "source_kl_lag_map" in out
    assert "te_lag_map" not in out
    assert not any("te_lag" in key for key in out)
