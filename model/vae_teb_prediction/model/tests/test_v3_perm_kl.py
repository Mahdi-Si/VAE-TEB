r"""S3-T03: the source-permutation control (G6).

**Batch-independence of the source path.** ``perm_kl_from_forward`` permutes the *already
computed* ``source_state`` instead of re-encoding a permuted source stream. That is only
legitimate because ``source_adapter`` and ``source_encoder`` contain no batch-coupled
operator, i.e. :math:`\mathrm{SourceEncoder}(\mathrm{SourceAdapter}(\pi(U)))_i = H_u[\pi(i)]`.
The whole DDP-safety argument for fusing the control into the single main backward rests on
this identity, so it is asserted directly and again end-to-end.

**Gradient routing.** With ``detach_prior=True`` the control cannot be minimised by dragging
the prior toward :math:`q`; only the source encoder, the attention, and the posterior deltas
may move.

**Behaviour, and a correction to the sprint's premise.** The spec expected a trained model to
shed most of its KL under a deranged source. It does not. On a seeded micro-overfit whose
future target is a lagged function of the source -- a model that unambiguously *uses* the
source, cutting forecast error from 14.20 (target-only) to 3.87 -- the deranged-source KL comes
out *above* the true one (6.62 vs 6.04). :math:`\mathrm{KL}(q\|p)` measures "the source moved
my belief", not "...correctly"; a wrong source is still a source, and the posterior reacts to
it out of distribution.

So the tests below pin the claims that survive: the model earns :math:`K_{\mathrm{true}}`; a
deranged source *destroys the forecast* (26.32, worse than using no source at all) -- the
negative control that genuinely discriminates; and, as a characterisation test,
:math:`K_{\mathrm{raw}}` does **not** separate. That last one exists so nobody later reads
``kld_raw`` as source-specific without re-deriving the result.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
    SeqVaeLagAttnV3,
    make_derangement,
)

_PROD_FLAGS = dict(posterior_logvar="residual", logvar_bound="smooth", kld_support="anchor")


def _perturb_posterior(model: SeqVaeLagAttnV3, seed: int = 3) -> None:
    """Break the zero-init so the permuted and true posteriors actually differ.

    At initialisation both delta heads output exactly 0, so every KL in this module would be
    0 and the equivalence assertions would pass vacuously.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.posterior_head.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.1)


@pytest.mark.parametrize("head_structured", [False, True])
def test_permuting_source_state_equals_re_encoding_permuted_source(
    tiny_kwargs, inputs, head_structured
):
    """The identity that licenses the fused control."""
    model = SeqVaeLagAttnV3(
        head_structured_latent=head_structured, **_PROD_FLAGS, **tiny_kwargs
    ).eval()
    y_st, y_ph, u = inputs
    perm = make_derangement(y_st.size(0), generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        h_u = model(*inputs)["source_state"]
        h_u_reencoded = model.source_encoder(model.source_adapter(u[perm]))

    # float32 round-off through the conv/LSTM stack, not a structural difference.
    deviation = (h_u[perm] - h_u_reencoded).abs().max().item()
    scale = h_u.abs().max().item()
    assert deviation < 1e-5 * max(scale, 1.0), (
        f"source path is not batch-independent (max deviation {deviation:.3e}, "
        f"activation scale {scale:.3e}); the fused permutation control is invalid"
    )


@pytest.mark.parametrize("head_structured", [False, True])
def test_fused_perm_kl_matches_re_encoded_perm_kl(tiny_kwargs, inputs, head_structured):
    model = SeqVaeLagAttnV3(
        head_structured_latent=head_structured, **_PROD_FLAGS, **tiny_kwargs
    ).eval()
    _perturb_posterior(model)
    perm = make_derangement(inputs[0].size(0), generator=torch.Generator().manual_seed(1))

    fused = model.perm_kl_from_forward(model(*inputs), perm_index=perm)
    exact = model.permutation_kl(*inputs, perm_index=perm)

    # Guard against a vacuous pass: the perturbed posterior must produce a real KL.
    assert float(exact["perm_kl"]) > 1e-3, "perm_kl is ~0; the equivalence check is vacuous"
    assert torch.allclose(fused["perm_kl"], exact["perm_kl"], atol=1e-6)
    assert torch.equal(fused["perm_index"], exact["perm_index"])
    assert not fused["kld_shuffled"].requires_grad


def test_perm_kl_respects_a_supplied_weight(tiny_kwargs, inputs):
    """A uniform ``(B, T)`` weight cancels between numerator and denominator."""
    model = SeqVaeLagAttnV3(**_PROD_FLAGS, **tiny_kwargs).eval()
    _perturb_posterior(model)
    perm = make_derangement(2, generator=torch.Generator().manual_seed(2))
    B, T = inputs[0].shape[:2]

    unweighted = model.permutation_kl(*inputs, perm_index=perm)["perm_kl"]
    doubled = model.permutation_kl(
        *inputs, perm_index=perm, weight=torch.full((B, T), 2.0)
    )["perm_kl"]
    assert torch.allclose(unweighted, doubled, atol=1e-6)

    # A weight that zeroes one sample must change the mean.
    lopsided = torch.ones(B, T)
    lopsided[0] = 0.0
    assert not torch.allclose(
        unweighted, model.permutation_kl(*inputs, perm_index=perm, weight=lopsided)["perm_kl"]
    )


def _grads(module) -> list:
    return [p.grad for p in module.parameters() if p.grad is not None]


def test_detach_prior_routes_gradient_away_from_the_prior(tiny_kwargs, inputs):
    r"""With the prior detached, only the source/posterior path may move."""
    model = SeqVaeLagAttnV3(**_PROD_FLAGS, **tiny_kwargs)
    _perturb_posterior(model)
    perm = make_derangement(2, generator=torch.Generator().manual_seed(4))

    model.zero_grad(set_to_none=True)
    model.permutation_kl(*inputs, perm_index=perm, detach_prior=True)["perm_kl"].backward()
    assert not _grads(model.prior_head), (
        "detach_prior=True still let L_perm move the prior; the control could then be "
        "minimised by dragging p toward q instead of collapsing the source-driven deltas"
    )
    assert _grads(model.source_encoder), "source encoder got no gradient"
    assert _grads(model.lag_attn), "lag attention got no gradient"
    assert _grads(model.posterior_head.delta_mu_head), "delta_mu_head got no gradient"
    # The decoders are untouched by the control (they are not in its graph at all).
    assert not _grads(model.baseline_decoder)
    assert not _grads(model.residual_decoder)

    model.zero_grad(set_to_none=True)
    model.permutation_kl(*inputs, perm_index=perm, detach_prior=False)["perm_kl"].backward()
    assert _grads(model.prior_head), "detach_prior=False must let the prior receive gradient"


def test_perm_kl_is_zero_at_init(tiny_kwargs, inputs):
    """At init the deltas vanish, so even a shuffled source produces no KL."""
    model = SeqVaeLagAttnV3(**_PROD_FLAGS, **tiny_kwargs).eval()
    perm = make_derangement(2, generator=torch.Generator().manual_seed(5))
    assert float(model.permutation_kl(*inputs, perm_index=perm)["kld_shuffled"]) < 1e-6


def test_perm_kl_rejects_a_degenerate_batch(tiny_kwargs):
    model = SeqVaeLagAttnV3(**_PROD_FLAGS, **tiny_kwargs).eval()
    y_st, y_ph = torch.randn(1, 16, 43), torch.randn(1, 16, 44)
    u = torch.randn(1, 16, 101)
    with pytest.raises(ValueError, match="batch_size >= 2"):
        model.permutation_kl(y_st, y_ph, u)


# ---------------------------------------------------------------------------
# Behavioural: seeded micro-overfit on a source-coupled task
# ---------------------------------------------------------------------------
_MICRO_KWARGS = dict(
    sequence_length=24, d_model=32, d_z=8, horizon=4, warmup_period=4,
    c_y=87, c_u=58, use_up_st=False, max_lag=8, num_heads=4, d_head=8, dropout=0.0,
    causal_norm=True,
)
_LAG = 5     # driver delay; exceeds the horizon so Y_{<=t} cannot supply the needed innovations
_RHO = 0.9   # driver autocorrelation
_BATCH, _SEQ = 8, 24
_MIXING = torch.randn(87, generator=torch.Generator().manual_seed(777)) * 1.5


def _source_coupled_batch(seed: int):
    r"""Sample ``(y_st, y_ph, u)`` from :math:`Y_t = A\,d_{t-5} + \varepsilon`, :math:`U_{t,0} = d_t`.

    The driver :math:`d` is AR(1). :math:`Y_{\le t}` reveals :math:`d_{\le t-5}`, from which a
    causal target-only baseline can extrapolate part of :math:`d_{t-4..t-1}`; the innovations
    it cannot reach live only in :math:`U_{\le t}`. Transfer entropy from :math:`U` to :math:`Y`
    is therefore strictly positive and the source is the only route to a good forecast.

    Two design choices matter. ``causal_norm=True``: v1's encoders pool ``GroupNorm``
    statistics across time, so a "causal" baseline can read the future and needs no source at
    all. **Fresh batches every step**: with one fixed batch the target history uniquely
    identifies the sample, so the model memorises its own future and again needs no source.
    """
    g = torch.Generator().manual_seed(seed)
    innovations = torch.randn(_BATCH, _SEQ, generator=g)
    driver = torch.zeros(_BATCH, _SEQ)
    for t in range(1, _SEQ):
        driver[:, t] = _RHO * driver[:, t - 1] + innovations[:, t]

    u = torch.randn(_BATCH, _SEQ, 58, generator=g)
    u[:, :, 0] = driver
    delayed = torch.zeros_like(driver)
    delayed[:, _LAG:] = driver[:, :-_LAG]
    y = delayed.unsqueeze(-1) * _MIXING + 0.1 * torch.randn(_BATCH, _SEQ, 87, generator=g)
    return y[..., :43].contiguous(), y[..., 43:].contiguous(), u


def _kl_controls(model, y_st, y_ph, u, n_perms: int = 3):
    """``(K_true, K_shuffled)`` on the shared anchor support."""
    k_true = float(model.measure_transfer_entropy(y_st, y_ph, u, reduce_mean=True))
    shuffled = [
        float(model.permutation_kl(
            y_st, y_ph, u,
            perm_index=make_derangement(
                y_st.size(0), generator=torch.Generator().manual_seed(100 + i)
            ),
        )["kld_shuffled"])
        for i in range(n_perms)
    ]
    return k_true, sum(shuffled) / len(shuffled)


def _forecast_controls(model, y_st, y_ph, u, n_perms: int = 3):
    """``(feat_loss, base_loss, feat_loss_shuffled)`` -- the prediction-space control."""
    model.eval()
    with torch.no_grad():
        outs = model(y_st, y_ph, u)
        losses = model.compute_loss(outs, y_st, y_ph, beta=0.0, likelihood="mse")
        shuffled = []
        for i in range(n_perms):
            perm = make_derangement(
                y_st.size(0), generator=torch.Generator().manual_seed(200 + i)
            )
            permuted = model.perm_forward_outputs(outs, perm_index=perm)
            shuffled.append(float(model.compute_loss(
                permuted, y_st, y_ph, beta=0.0, likelihood="mse", compute_kld_loss=False
            )["feat_loss"]))
    return (
        float(losses["feat_loss"]),
        float(losses["base_loss"]),
        sum(shuffled) / len(shuffled),
    )


@pytest.fixture(scope="module")
def trained_model() -> SeqVaeLagAttnV3:
    """Train once (~40 s) and share across the three behavioural tests below."""
    steps, seed = 300, 0
    torch.manual_seed(seed)
    model = SeqVaeLagAttnV3(**_PROD_FLAGS, **_MICRO_KWARGS)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    model.train()
    for step in range(steps):
        y_st, y_ph, u = _source_coupled_batch(1000 + step)  # a fresh batch every step
        opt.zero_grad(set_to_none=True)
        outs = model(y_st, y_ph, u)
        model.compute_loss(
            outs, y_st, y_ph, beta=1e-3, free_bits=0.0, likelihood="mse"
        )["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        model.train()
    return model


def test_micro_overfit_learns_to_use_the_source(trained_model):
    r"""The model earns :math:`K_{\mathrm{true}}` and the source genuinely drives the forecast.

    Empirical at a fixed seed, on held-out data; the thresholds are margins, not analytic
    bounds. Measured at seed 0: ``K_true`` 6.04, ``feat`` 3.87, ``base`` 14.20.
    """
    torch.manual_seed(0)
    fresh = SeqVaeLagAttnV3(**_PROD_FLAGS, **_MICRO_KWARGS)
    y_st, y_ph, u = _source_coupled_batch(99)
    k_true_0, k_shuf_0 = _kl_controls(fresh, y_st, y_ph, u)
    assert k_true_0 < 1e-6 and k_shuf_0 < 1e-6, "zero-KL init violated before training"

    k_true, _ = _kl_controls(trained_model, y_st, y_ph, u)
    feat, base, _ = _forecast_controls(trained_model, y_st, y_ph, u)

    assert k_true > 0.5, f"the model earned almost no KL: K_true={k_true:.4e}"
    assert feat < 0.5 * base, (
        f"the source bought no forecast skill: feat={feat:.3f}, base={base:.3f}"
    )


def test_a_deranged_source_destroys_the_forecast(trained_model):
    r"""The negative control that actually discriminates.

    A model that exploits the source satisfies ``feat < base < feat_shuffled``: a *wrong*
    source is worse than no source at all. Measured at seed 0: 3.87 < 14.20 < 26.32.
    """
    y_st, y_ph, u = _source_coupled_batch(99)
    feat, base, feat_shuffled = _forecast_controls(trained_model, y_st, y_ph, u)

    assert feat < base, f"feat={feat:.3f} is not below base={base:.3f}"
    assert feat_shuffled > base, (
        f"a deranged source is not worse than no source: "
        f"feat_shuffled={feat_shuffled:.3f}, base={base:.3f}"
    )
    assert feat_shuffled > 2.0 * feat


def test_raw_kl_does_not_separate_under_a_deranged_source(trained_model):
    r"""Characterisation test: :math:`K_{\mathrm{raw}}` is **not** a source-specific statistic.

    :math:`\mathrm{KL}(q\|p)` measures "the source moved my belief", not "...*correctly*". A
    deranged UP is still a UP, and the posterior -- trained only on matched pairs -- reacts to
    it out of distribution, typically *more* strongly. So :math:`K_{\mathrm{shuffled}} \gtrsim
    K_{\mathrm{true}}` even for the model of
    :func:`test_a_deranged_source_destroys_the_forecast`, which demonstrably uses the source.

    This is why ``lambda_perm`` ships at ``0.0`` (readout only) and why the prediction-space
    control above is the one to trust. Measured ratios at ``lambda_perm=0``: 1.10 (seed 0),
    1.02 (seed 1). Adding the control to the loss narrows the ratio to ~0.85 but destroyed the
    source pathway outright in 2 of 4 seeds.

    If this assertion ever fails, the KL *has* become source-specific -- a real result. Update
    the spec's S3 finding and the ``lambda_perm`` default rather than deleting the test.
    """
    y_st, y_ph, u = _source_coupled_batch(99)
    k_true, k_shuffled = _kl_controls(trained_model, y_st, y_ph, u)

    assert k_true > 0.5
    assert k_shuffled > 0.8 * k_true, (
        f"K_shuffled ({k_shuffled:.4f}) fell well below K_true ({k_true:.4f}); the raw KL now "
        "appears source-specific. See this test's docstring before changing it."
    )
