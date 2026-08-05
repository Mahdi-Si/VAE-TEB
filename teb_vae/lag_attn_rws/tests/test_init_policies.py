r"""The zero-parameter initialisation policies, and the contracts they must leave intact.

Three post-init re-initialisations shape the model's starting point without adding a parameter:
the horizon-embedding scale (so the horizon tokens are not near-identical at init), the output-head
calibration (so the raw-target NLL starts at the trivial predictor's level, not orders of magnitude
above it), and the posterior source gain (so the attended source summary is not out-columned by the
target state in the fusion). Each is a config key the init-off sweep arm reverts together, and each
default is an exact no-op -- which is why the policy-on assertions here build with the policy on
rather than merely constructing the model.

At init the KL is identically zero, so the KL/lag-map assertions perturb the posterior first (via
the shared ``perturb_posterior`` fixture); without that they pass on any model at all.
"""
from __future__ import annotations

import math

import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC, SeqVaeLagAttnRws


def _model(kwargs, **overrides) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**dict(kwargs, **overrides))


def _horizon_token_correlation(model, batch: int = 256, seed: int = 0) -> float:
    r"""Mean pairwise Pearson correlation among the $H$ horizon tokens entering the refine stack.

    The tokens are $h + e_k$ for a shared broadcast latent $h$ and per-step embeddings $e_k$; the
    embedding scale is exactly what ``horizon_embed_std`` sets, so this reads off whether the tokens
    start near-identical (small embedding) or distinct (large embedding), before any convolution.
    """
    core = model.horizon_core
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, 1, model.d_z, generator=generator)
    with torch.no_grad():
        h = model.decoder.proj(z)  # (batch, 1, d_hidden) -- the only latent entry point
        feat = h.unsqueeze(2).expand(-1, -1, core.horizon, -1) + core.horizon_embedding
        tokens = feat.reshape(batch, core.horizon, core.d_hidden)
        centered = tokens - tokens.mean(dim=-1, keepdim=True)
        unit = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        corr = unit @ unit.transpose(-1, -2)  # (batch, H, H)
        horizon = core.horizon
        off_diagonal = (corr.sum(dim=(-1, -2)) - horizon) / (horizon * (horizon - 1))
    return float(off_diagonal.mean())


def _per_sample_gaussian_nll(mu, logvar, target) -> torch.Tensor:
    r"""The model's own factorized-Gaussian per-sample NLL, its full $\log 2\pi$ constant included.

    Matches ``nets/losses.py::masked_raw_block_per_anchor`` so the number compared below is the one
    the objective actually sees; the constant is in both the calibrated and the trivial score, so
    the ratio the calibration is judged on does not depend on it.
    """
    return (
        0.5 * (math.log(2.0 * math.pi) + logvar + (target - mu) ** 2 * torch.exp(-logvar))
    ).mean()


# =========================================================================================
# Horizon-embedding std (R3-C1)
# =========================================================================================
def test_the_embedding_is_reseeded_at_the_configured_std(tiny_kwargs):
    std = float(_model(tiny_kwargs, horizon_embed_std=0.8).horizon_core.horizon_embedding.std())
    assert 0.7 < std < 0.9, f"embedding std {std} is not near the configured 0.8"


def test_the_default_std_leaves_the_core_seed_untouched(tiny_kwargs):
    """0.02 is the core's own seed, so the default policy is a no-op and the embedding stays small."""
    std = float(_model(tiny_kwargs, horizon_embed_std=0.02).horizon_core.horizon_embedding.std())
    assert std < 0.05


def test_a_large_std_breaks_the_horizon_token_symmetry(tiny_kwargs):
    """Self-verifying: the large std drops the token correlation well below the small-std negative
    control, so a refactor that dropped the reseed would fail here rather than pass silently."""
    corr_big = _horizon_token_correlation(_model(tiny_kwargs, horizon_embed_std=0.8))
    corr_small = _horizon_token_correlation(_model(tiny_kwargs, horizon_embed_std=0.02))

    assert corr_big < 0.9, f"tokens still {corr_big:.3f} correlated at std 0.8"
    assert corr_small > 0.95, f"control tokens only {corr_small:.3f} correlated at std 0.02"


def test_the_embedding_reseed_preserves_the_zero_kl_start(tiny_kwargs, inputs):
    """The policy touches only the decoder embedding, not the posterior deltas, so the KL is still
    exactly zero and the two forecasts bitwise identical at init."""
    model = _model(tiny_kwargs, horizon_embed_std=0.8).train()
    torch.manual_seed(0)
    out = model(*inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["mu_base"], out["mu_full"])


# =========================================================================================
# Output-head calibration (R3-C2)
# =========================================================================================
def test_calibrated_heads_start_at_the_trivial_predictor(tiny_kwargs):
    model = _model(tiny_kwargs, head_init_calibration=True)
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(4, 8, model.d_z, generator=generator)
    target = torch.randn(4, 8, model.horizon, model.raw_per_step, generator=generator)

    with torch.no_grad():
        mu, logvar = model.decoder(z)
    nll = _per_sample_gaussian_nll(mu, logvar, target)
    trivial = _per_sample_gaussian_nll(
        torch.zeros_like(target), torch.zeros_like(target), target
    )

    assert float(logvar.mean().abs()) < 0.1, "decoder log-variance is not centred at 0"
    assert float(mu.pow(2).mean().sqrt()) < 0.1, "decoder mean is not near 0"
    assert abs(float(nll) - float(trivial)) < 0.1 * float(trivial), (
        f"init NLL {float(nll):.3f} not within 10% of the trivial predictor {float(trivial):.3f}"
    )


def test_uncalibrated_heads_start_far_above_the_trivial_predictor(tiny_kwargs):
    """The negative control: Xavier-filled heads sit well above the trivial predictor -- exactly
    the init pressure the calibration removes."""
    model = _model(tiny_kwargs, head_init_calibration=False)
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(4, 8, model.d_z, generator=generator)
    target = torch.randn(4, 8, model.horizon, model.raw_per_step, generator=generator)

    with torch.no_grad():
        mu, logvar = model.decoder(z)
    nll = _per_sample_gaussian_nll(mu, logvar, target)
    trivial = _per_sample_gaussian_nll(
        torch.zeros_like(target), torch.zeros_like(target), target
    )

    assert float(nll) > 3.0 * float(trivial)


def test_the_logvar_bias_is_the_exact_preimage_of_zero_logvar(tiny_kwargs):
    """log(5/3) is the exact pre-image of log-variance 0 under smooth_bound(-5, 3): sigma = 1 in
    z-scored units, which is the trivial predictor's variance."""
    model = _model(tiny_kwargs, head_init_calibration=True)
    bias = model.decoder.logvar_head.bias
    lo, hi = model.logvar_clamp

    assert torch.allclose(smooth_bound(bias, lo, hi), torch.zeros_like(bias), atol=1e-6)


def test_calibration_keeps_the_perturbation_test_meaningful(
    tiny_kwargs, inputs, perturb_posterior
):
    """The mean head is scaled by 0.02, not zeroed, so a perturbed posterior still moves the two
    forecasts apart -- the ``mu_base != mu_full`` assertion in test_zero_kl_init stays non-vacuous
    under calibration."""
    model = _model(tiny_kwargs, head_init_calibration=True).train()
    perturb_posterior(model)
    torch.manual_seed(0)
    out = model(*inputs)

    assert not torch.equal(out["mu_base"], out["mu_full"])


def test_the_calibrated_prior_starts_at_unit_scale(tiny_kwargs, inputs):
    """The prior half of the calibration: the log-variance head's final layer and skip are
    zeroed and the bias seeded at the pre-image of 0, so the bounded output is exactly 0
    (sigma_p = 1, the scale anchor's optimum) for every input. Exactness is the point --
    smooth_bound is a sigmoid, so a merely shrunk head would start near zero only on average,
    not per coordinate. 1e-6 is float rounding on the log(5/3) -> sigmoid round trip, not a
    modelling tolerance."""
    model = _model(tiny_kwargs, head_init_calibration=True).eval()
    with torch.no_grad():
        out = model(*inputs)
    logvar_prior = out["logvar_prior"]

    assert float(logvar_prior.abs().max()) < 1e-6
    # The pinned-prior watch metric therefore opens at exactly zero: no coordinate is within
    # the floor margin of the clamp's lower end.
    lo, hi = model.logvar_clamp
    floor = lo + LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    assert float((logvar_prior <= floor).float().mean()) == 0.0


def test_the_uncalibrated_prior_is_not_at_unit_scale(tiny_kwargs, inputs):
    """The negative control: Xavier-filled, the head's raw output sits near 0 and the sigmoid
    bound maps it around -1, so the calibrated assertion above is not vacuous."""
    model = _model(tiny_kwargs, head_init_calibration=False).eval()
    with torch.no_grad():
        out = model(*inputs)

    assert float(out["logvar_prior"].abs().mean()) > 0.5


def test_the_prior_calibration_preserves_the_zero_kl_start(tiny_kwargs, inputs):
    """The posterior's log-variance residual is built on the prior's raw pre-bound tensor, so
    pinning that tensor moves prior and posterior together and the KL stays exactly zero."""
    model = _model(tiny_kwargs, head_init_calibration=True).train()
    torch.manual_seed(0)
    out = model(*inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


# =========================================================================================
# Posterior source gain (R2-C1)
# =========================================================================================
def test_the_a_head_gain_is_the_configured_constant(tiny_kwargs):
    weight = _model(tiny_kwargs, a_head_gain=2.0).posterior_head.a_head_norm.weight
    assert torch.equal(weight, torch.full_like(weight, 2.0))


def test_the_default_gain_is_the_plain_unit_norm(tiny_kwargs):
    weight = _model(tiny_kwargs, a_head_gain=1.0).posterior_head.a_head_norm.weight
    assert torch.equal(weight, torch.ones_like(weight))


def test_the_source_gain_preserves_the_zero_kl_start(tiny_kwargs, inputs):
    """The gain rescales the attended summary, but the posterior deltas are still zero at init, so
    the closed-form KL is exactly zero."""
    model = _model(tiny_kwargs, a_head_gain=2.0).train()
    torch.manual_seed(0)
    out = model(*inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0


# =========================================================================================
# Bundle closure: the structural contracts under the full shipped flag set
# =========================================================================================
def _shipped_flag_model(tiny_kwargs, **overrides) -> SeqVaeLagAttnRws:
    """The tiny geometry with the production init flags on (per-block FiLM is already hardcoded),
    so the contracts are proven for the architecture the study runs, not only for the defaults."""
    return _model(
        tiny_kwargs,
        horizon_embed_std=0.8,
        head_init_calibration=True,
        a_head_gain=2.0,
        **overrides,
    )


def test_the_full_shipped_flag_set_starts_at_exactly_zero_kl(tiny_kwargs, inputs):
    model = _shipped_flag_model(tiny_kwargs).train()
    torch.manual_seed(0)
    out = model(*inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


def test_the_full_shipped_flag_set_preserves_the_lag_map_identity(
    tiny_kwargs, inputs, perturb_posterior
):
    """Perturbed, so it is not the vacuous zero map: under the full production init the lag
    attribution still sums over lags to the per-step KL exactly."""
    model = _shipped_flag_model(tiny_kwargs).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)

    assert float(out["kld_per_t"].abs().max()) > 0.0, "perturbation failed; test is vacuous"
    total = out["source_kl_lag_map"].sum(dim=-1)
    assert torch.allclose(total, out["kld_per_t"], atol=1e-5, rtol=1e-5)
