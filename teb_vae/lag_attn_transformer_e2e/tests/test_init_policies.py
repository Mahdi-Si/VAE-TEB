r"""The initialisation policies this model applies after its module tree is built.

The generic per-layer-type pass xavier-fills every ``nn.Linear`` and every ``nn.Conv1d``; what runs
after it is load-bearing:

1. the variance-preserving depthwise correction, which repairs what that pass does to a
   $(C, 1, k)$ filter bank -- Xavier reads $\mathrm{fan\_in} = k$ against
   $\mathrm{fan\_out} = Ck$, a factor $\sqrt{(1+C)/2}$ too small and independent of $k$;
2. the per-block FiLM re-zeroing, which the generic pass undoes;
3. the three zero-parameter calibration policies, each applied only when its configured value
   leaves the constructor default.

Step 1 is where this package's risk lives. The front ends contribute one more depthwise filter bank
per stage per stream than the sibling has, and the repair pass finds them only because they are the
sibling's ``CausalDepthwiseConv1d``, the exact class it detects. So both the count and the standard
deviation are measured.

The posterior-delta zeroing is asserted behaviourally, as the exact zero-KL start, in
``test_zero_kl_init.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC
from teb_vae.lag_attn_transformer_e2e.nets.frontend import NUM_STAGES
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import SHIPPED_KWARGS
from teb_vae.lag_attn_transformer_rws.nets.blocks import CausalDepthwiseConv1d

#: Fractional band the measured depthwise standard deviation must sit inside.
#:
#: Wider than the sibling's $10\%$, and the reason is the *number* of banks rather than their width:
#: the assertion below must hold for every stem and front-end convolution at once, and the worst
#: relative deviation of the measured $\sigma$ from $1/\sqrt{k}$ across twenty-five seeds reaches
#: $9.7\%$. A $10\%$ band would flake; $20\%$ does not, and is still far tighter than the error
#: this test exists to catch.
_STD_BAND = 0.20


def _model(kwargs, **overrides) -> SeqVaeLagAttnTrfE2E:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfE2E(**dict(kwargs, **overrides))


@pytest.fixture(scope="module")
def shipped_model() -> SeqVaeLagAttnTrfE2E:
    """One production-geometry model, built once: the depthwise arithmetic is only separated from
    the generic pass by a wide enough margin at the shipped widths to assert on."""
    return _model(SHIPPED_KWARGS)


# =========================================================================================
# The depthwise correction
# =========================================================================================
def test_the_depthwise_pass_reinitialised_the_front_ends_as_well_as_the_stems(shipped_model):
    """The count this package adds, stated as its two contributions: one convolution per stem
    kernel per stream, and one per front-end stage per stream."""
    stems = 2 * len(SHIPPED_KWARGS["encoder_conv_kernels"])
    frontends = 2 * NUM_STAGES

    assert shipped_model.n_depthwise_init == stems + frontends


def test_depthwise_weights_carry_the_variance_preserving_scale(shipped_model):
    r"""$\sigma = 1/\sqrt{k}$, which is what preserves the variance of a $k$-term sum.

    Asserted over every depthwise convolution in the model, front ends included: a wrong standard
    deviation is what the count above cannot see, and the front end's widest first-stage kernel is
    where it would matter most. A correction that ran before the generic pass, or not at all, would
    leave Xavier's far smaller value here.
    """
    convolutions = [
        module for module in shipped_model.modules()
        if isinstance(module, CausalDepthwiseConv1d)
    ]
    assert len(convolutions) == shipped_model.n_depthwise_init

    for convolution in convolutions:
        target = 1.0 / math.sqrt(float(convolution.kernel_size))
        measured = float(convolution.conv.weight.std())
        assert abs(measured - target) < _STD_BAND * target, (
            f"kernel {convolution.kernel_size}: measured std {measured:.4f} is not within "
            f"{_STD_BAND:.0%} of the variance-preserving {target:.4f}"
        )


# =========================================================================================
# The FiLM re-zeroing the generic pass would have undone
# =========================================================================================
def test_the_film_generators_are_exactly_zero(tiny_kwargs):
    model = _model(tiny_kwargs)
    film = model.horizon_core.refine.film

    assert film is not None and len(film) > 0
    for generator in film:
        assert float(generator.weight.abs().max()) == 0.0


# =========================================================================================
# The three calibration policies
# =========================================================================================
def test_the_horizon_embedding_is_reseeded_at_the_configured_std(tiny_kwargs):
    std = float(_model(tiny_kwargs, horizon_embed_std=0.8).horizon_core.horizon_embedding.std())
    assert 0.7 < std < 0.9, f"embedding std {std} is not near the configured 0.8"


def test_the_logvar_bias_is_the_exact_preimage_of_zero_logvar(tiny_kwargs):
    r"""$\log(5/3)$ is the exact pre-image of log-variance $0$ under ``smooth_bound(-5, 3)``:
    $\sigma = 1$ in z-scored units, which is the trivial predictor's variance. Without it the
    raw-target NLL starts about $15$ nats per sample above that predictor, and the first epochs
    measure the optimiser undoing the initialisation rather than either input representation."""
    model = _model(tiny_kwargs, head_init_calibration=True)
    bias = model.decoder.logvar_head.bias
    lo, hi = model.logvar_clamp

    assert float(bias.min()) == pytest.approx(math.log(5.0 / 3.0))
    assert torch.allclose(smooth_bound(bias, lo, hi), torch.zeros_like(bias), atol=1e-6)


def test_the_calibrated_prior_starts_at_unit_scale(tiny_kwargs, raw_inputs):
    """The prior half of the calibration: the log-variance head's final layer and skip are zeroed and
    the bias seeded at the pre-image of 0, so the bounded output is exactly 0 (sigma_p = 1, the scale
    anchor's optimum) for every input. Exactness is the point -- smooth_bound is a sigmoid, so a
    merely shrunk head would start near zero only on average, not per coordinate. 1e-6 is float
    rounding on the log(5/3) -> sigmoid round trip, not a modelling tolerance."""
    model = _model(tiny_kwargs, head_init_calibration=True).eval()
    with torch.no_grad():
        out = model(*raw_inputs)
    logvar_prior = out["logvar_prior"]

    assert float(logvar_prior.abs().max()) < 1e-6
    # The pinned-prior watch metric therefore opens at exactly zero: no coordinate is within the
    # floor margin of the clamp's lower end.
    lo, hi = model.logvar_clamp
    floor = lo + LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    assert float((logvar_prior <= floor).float().mean()) == 0.0


def test_the_a_head_gain_reaches_the_posterior_fusion(tiny_kwargs):
    weight = _model(tiny_kwargs, a_head_gain=2.0).posterior_head.a_head_norm.weight
    assert torch.equal(weight, torch.full_like(weight, 2.0))


# =========================================================================================
# The bundle, under the shipped flag set
# =========================================================================================
def _shipped_flag_model(tiny_kwargs, **overrides) -> SeqVaeLagAttnTrfE2E:
    """The tiny geometry with the production init flags on, so the contracts below are proven for
    the architecture that trains rather than for the constructor defaults."""
    return _model(
        tiny_kwargs,
        horizon_embed_std=0.8,
        head_init_calibration=True,
        a_head_gain=2.0,
        **overrides,
    )


def test_the_full_shipped_flag_set_starts_at_exactly_zero_kl(tiny_kwargs, raw_inputs):
    """The posterior's log-variance residual is built on the prior's raw pre-bound tensor, so the
    prior calibration moves prior and posterior together and the KL stays exactly zero."""
    model = _shipped_flag_model(tiny_kwargs).train()
    torch.manual_seed(0)
    out = model(*raw_inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["logvar_post"], out["logvar_prior"])
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


def test_the_calibration_still_lets_a_perturbed_posterior_move_the_forecasts(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    """The mean head is *scaled* by $0.02$, not zeroed, so the two forecasts still separate under a
    perturbed posterior -- which is what keeps the zero-KL suite's ``mu_base != mu_full`` control
    non-vacuous under calibration."""
    model = _shipped_flag_model(tiny_kwargs).train()
    perturb_posterior(model)
    torch.manual_seed(0)
    out = model(*raw_inputs)

    assert not torch.equal(out["mu_base"], out["mu_full"])
