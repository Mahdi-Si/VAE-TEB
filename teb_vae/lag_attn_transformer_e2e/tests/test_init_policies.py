r"""The initialisation order, and every policy that depends on it.

Five things happen after the module tree is built, and the order is load-bearing rather than
stylistic:

1. the generic per-layer-type pass, which xavier-fills every ``nn.Linear`` and every ``nn.Conv1d``;
2. the variance-preserving depthwise correction, which repairs what that pass does to a
   $(C, 1, k)$ filter bank -- Xavier reads $\mathrm{fan\_in} = k$ against
   $\mathrm{fan\_out} = Ck$, a factor $\sqrt{(1+C)/2} = 8.03$ too small at $C = 128$ and
   independent of $k$, so no kernel sweep could reveal it;
3. the posterior delta zeroing, which the generic pass would otherwise have undone -- and with it
   the exact zero-KL start;
4. the per-block FiLM re-zeroing, which the generic pass also undoes;
5. the three zero-parameter calibration policies, each applied only when its configured value
   leaves the constructor default.

Step 2 is where this package's risk actually lives, and it is why this file exists rather than being
inherited by assertion. The front ends contribute **eight** more depthwise filter banks than the
sibling has -- one per stage per stream -- and the repair pass finds them only because they are the
sibling's ``CausalDepthwiseConv1d``, the exact class it detects. Had the front end grown a
convolution of its own the count would still have looked plausible, every shape test would still
have passed, and both front ends would have trained an order of magnitude too quiet with no signal
anywhere. A count alone would not have caught a wrong standard deviation either, so both are
measured.

Two module families are *not* touched by the generic pass, and that is what makes the pre-norm stack
start near the identity: ``LayerScale`` and ``RMSNorm`` hold bare ``nn.Parameter`` tensors on custom
modules rather than ``nn.Linear`` weights or ``nn.LayerNorm`` scales. Asserting their values alone
would pass on a generic pass that started handling them, so the negative control below runs the pass
a second time and requires it to move a linear weight while leaving these two untouched.

At init the KL is identically zero, so every KL assertion here builds with the policy on and, where
it must be non-vacuous, perturbs the posterior first.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import initialization, smooth_bound
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC
from teb_vae.lag_attn_transformer_e2e.nets.frontend import NUM_STAGES
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import SHIPPED_KWARGS
from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    LAYER_SCALE_INIT,
    CausalDepthwiseConv1d,
    LayerScale,
    RMSNorm,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import SHIPPED_KWARGS as SIBLING_SHIPPED

#: Fractional band the measured depthwise standard deviation must sit inside.
#:
#: Wider than the sibling's $10\%$, and the reason is the *number* of banks rather than their width.
#: At the shipped geometry the narrowest depthwise bank here is still the encoder stem's
#: $Ck = 640$, exactly the sibling's -- the front end's first stage is only a quarter of $d_{model}$
#: wide but carries a $65$-tap kernel, so it draws $2080$ samples. What changed is that the
#: assertion below must hold for **twelve** convolutions rather than four, and the worst relative
#: deviation of the measured $\sigma$ from $1/\sqrt{k}$ across twenty-five seeds reaches $9.7\%$.
#: A $10\%$ band would flake; $20\%$ does not, and is still far tighter than the $8\times$ error
#: this test exists to catch.
_STD_BAND = 0.20

#: How far above the generic pass's standard deviation the corrected one must sit for the correction
#: to be worth having. The predicted factor is $8.03$ at $C = 128$ and $4.1$ at the front end's
#: narrowest $C = 32$; $3$ leaves room for the sampling spread above without admitting a no-op.
_MIN_CORRECTION_FACTOR = 3.0


def _model(kwargs, **overrides) -> SeqVaeLagAttnTrfE2E:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfE2E(**dict(kwargs, **overrides))


def _xavier_std(weight: torch.Tensor) -> float:
    r"""The standard deviation ``xavier_uniform_`` produces on this exact weight shape.

    Computed here from the tensor rather than written down, so the comparison stays true if the
    shipped kernels change.
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    return math.sqrt(2.0 / float(fan_in + fan_out))


@pytest.fixture(scope="module")
def shipped_model() -> SeqVaeLagAttnTrfE2E:
    """One production-geometry model, built once: the depthwise arithmetic is only separated from
    the generic pass by a wide enough margin at the shipped widths to assert on.

    Reads :data:`SHIPPED_KWARGS` directly rather than the function-scoped fixture, so the build is
    paid for once for the whole module.
    """
    return _model(SHIPPED_KWARGS)


# =========================================================================================
# What the generic pass must leave alone
# =========================================================================================
def test_layer_scale_vectors_start_at_the_configured_gain(tiny_kwargs):
    model = _model(tiny_kwargs)
    scales = [module for module in model.modules() if isinstance(module, LayerScale)]

    assert scales, "no LayerScale in the model; the probe is vacuous"
    for scale in scales:
        assert torch.equal(scale.weight, torch.full_like(scale.weight, LAYER_SCALE_INIT))


def test_rms_norm_weights_start_at_one(tiny_kwargs):
    model = _model(tiny_kwargs)
    norms = [module for module in model.modules() if isinstance(module, RMSNorm)]

    assert norms, "no RMSNorm in the model; the probe is vacuous"
    for norm in norms:
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))


def test_the_generic_pass_moves_linears_and_leaves_the_two_alone(tiny_kwargs):
    """The negative control for the two tests above.

    Their values are also what a completely inert initialiser would leave, so the assertions only
    mean something if the pass under suspicion demonstrably *does* something. Running it a second
    time must move a linear weight -- proving it reached the model -- while both custom parameter
    families stay exactly where the constructor put them.
    """
    model = _model(tiny_kwargs)
    linear = model.query_proj.weight.detach().clone()
    scales = [module.weight.detach().clone() for module in model.modules()
              if isinstance(module, LayerScale)]
    norms = [module.weight.detach().clone() for module in model.modules()
             if isinstance(module, RMSNorm)]

    torch.manual_seed(11)
    initialization(model)

    assert not torch.equal(linear, model.query_proj.weight), (
        "the generic pass did not move a linear weight; it never reached the model"
    )
    for before, module in zip(
        scales, (m for m in model.modules() if isinstance(m, LayerScale))
    ):
        assert torch.equal(before, module.weight)
    for before, module in zip(
        norms, (m for m in model.modules() if isinstance(m, RMSNorm))
    ):
        assert torch.equal(before, module.weight)


def test_the_generic_pass_leaves_the_anti_alias_filters_bitwise_identical(tiny_kwargs):
    """The front end's own version of the same trap, and the reason its filters are buffers rather
    than ``nn.Conv1d`` weights: the generic pass Xavier-fills every convolution weight it walks, and
    a fixed kernel stored as a layer would be replaced by random values with no error and no symptom
    beyond aliased high-frequency energy appearing as real variability.

    Paired with a linear weight that the same pass must visibly move, so this cannot pass against an
    ``initialization`` that walked nothing at all.
    """
    model = _model(tiny_kwargs)
    filters = {name: buffer.clone() for name, buffer in model.named_buffers()
               if name.endswith("fir")}
    linear = model.query_proj.weight.detach().clone()

    assert len(filters) == 2 * NUM_STAGES, filters

    torch.manual_seed(11)
    initialization(model)

    assert not torch.equal(linear, model.query_proj.weight)
    for name, before in filters.items():
        after = dict(model.named_buffers())[name]
        assert torch.equal(before, after), f"{name} was overwritten by the generic pass"


# =========================================================================================
# The depthwise correction
# =========================================================================================
def test_the_depthwise_pass_reinitialised_the_front_ends_as_well_as_the_stems(shipped_model):
    """The count this package adds, stated as its two contributions. The stems are the sibling's --
    one per stream, one convolution per stem kernel -- and the front ends contribute one per stage
    per stream. Building the stages from the sibling's ``GatedCausalConvBlock`` is what makes them
    visible to the repair pass with no extension to it."""
    stems = 2 * len(SHIPPED_KWARGS["encoder_conv_kernels"])
    frontends = 2 * NUM_STAGES

    assert shipped_model.n_depthwise_init == stems + frontends == 12


def test_the_count_is_strictly_greater_than_the_siblings_at_equal_stem_settings(shipped_model):
    """The comparison that says the front ends were *included* rather than merely counted. Both
    models are built at the same stem schedule, so the entire difference is the front ends."""
    torch.manual_seed(0)
    sibling = SeqVaeLagAttnTrfRws(**SIBLING_SHIPPED)

    assert SHIPPED_KWARGS["encoder_conv_kernels"] == SIBLING_SHIPPED["encoder_conv_kernels"]
    assert shipped_model.n_depthwise_init > sibling.n_depthwise_init
    assert shipped_model.n_depthwise_init - sibling.n_depthwise_init == 2 * NUM_STAGES


def test_depthwise_weights_carry_the_variance_preserving_scale(shipped_model):
    r"""$\sigma = 1/\sqrt{k}$, which is what preserves the variance of a $k$-term sum.

    Asserted over every depthwise convolution in the model, front ends included: a wrong standard
    deviation is what the count above cannot see, and the front end's $65$-tap first stage is where
    it would matter most.
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


def test_the_correction_is_far_above_what_the_generic_pass_would_have_left(shipped_model):
    """The number the correction exists for: at $k = 5$, $C = 128$ the generic pass gives
    $\\sqrt{2/(5 + 640)} = 0.0557$ against a target of $1/\\sqrt 5 = 0.447$. The comparison is
    computed on the same weight shape rather than written down."""
    for convolution in shipped_model.modules():
        if not isinstance(convolution, CausalDepthwiseConv1d):
            continue
        generic = _xavier_std(convolution.conv.weight)
        measured = float(convolution.conv.weight.std())
        assert measured > _MIN_CORRECTION_FACTOR * generic, (
            f"kernel {convolution.kernel_size}: measured std {measured:.4f} is not "
            f"{_MIN_CORRECTION_FACTOR}x the generic pass's {generic:.4f} -- the depthwise "
            f"correction did not run, or ran before it"
        )


def test_the_correction_runs_after_the_generic_pass_not_before(tiny_kwargs):
    """Order, stated as a counterfactual: applying the generic pass to a built model undoes the
    depthwise scale, so a model whose correction ran first would look like this one. Measured on a
    front-end convolution specifically, since that is the stack the ordering was extended to cover.

    The **last** stage, at the full $d_{model}$: the damage Xavier does to a $(C, 1, k)$ bank is a
    factor $\\sqrt{(1+C)/2}$, so at the first stage's $C = d_{model}/4 = 8$ it is only $2.1\\times$
    and the halving below would be a coin flip on the sampling spread. The tapered width schedule is
    the reason, and it is also why this file asserts the *value* everywhere rather than relying on
    this counterfactual alone.
    """
    model = _model(tiny_kwargs)
    convolution = model.target_frontend.stage_modules[-1].block.conv
    corrected = float(convolution.conv.weight.std())

    initialization(model)
    undone = float(convolution.conv.weight.std())

    assert undone < 0.5 * corrected


# =========================================================================================
# The two zeroings the generic pass would have undone
# =========================================================================================
def test_the_posterior_delta_heads_are_exactly_zero(tiny_kwargs):
    model = _model(tiny_kwargs)

    for name in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(model.posterior_head, name)
        layers = list(module) if isinstance(module, nn.ModuleList) else [module]
        for layer in layers:
            assert float(layer.weight.abs().max()) == 0.0, f"{name} weight not zeroed"
            if layer.bias is not None:
                assert float(layer.bias.abs().max()) == 0.0, f"{name} bias not zeroed"


def test_the_film_generators_are_exactly_zero(tiny_kwargs):
    model = _model(tiny_kwargs)
    film = model.horizon_core.refine.film

    assert film is not None and len(film) > 0
    for generator in film:
        assert float(generator.weight.abs().max()) == 0.0


@pytest.mark.parametrize("init_weights", [True, False], ids=["generic-pass", "no-generic-pass"])
def test_without_the_rezeroing_neither_is_zero(tiny_kwargs, monkeypatch, init_weights):
    """The negative control for both, in the two directions they can fail.

    With the generic pass on, it xavier-refills the delta heads *and* the FiLM generators the horizon
    core zero-initialised itself -- that refill is precisely what the re-zeroing repairs. With it
    off, the delta heads keep torch's own non-zero default while the core's FiLM zeros survive, so
    only the delta assertion is testable there; that asymmetry is why the FiLM generators need the
    generic pass to have a control at all.
    """
    monkeypatch.setattr(SeqVaeLagAttnTrfE2E, "_zero_init_delta_heads", lambda self: None)
    monkeypatch.setattr(SeqVaeLagAttnTrfE2E, "_zero_init_film_generators", lambda self: None)
    model = _model(tiny_kwargs, init_weights=init_weights)

    assert float(model.posterior_head.delta_mu_head[0].weight.abs().max()) > 0.0
    if init_weights:
        film = model.horizon_core.refine.film
        assert film is not None
        assert max(float(generator.weight.abs().max()) for generator in film) > 0.0


# =========================================================================================
# The three calibration policies
# =========================================================================================
def test_the_horizon_embedding_is_reseeded_at_the_configured_std(tiny_kwargs):
    std = float(_model(tiny_kwargs, horizon_embed_std=0.8).horizon_core.horizon_embedding.std())
    assert 0.7 < std < 0.9, f"embedding std {std} is not near the configured 0.8"


def test_the_default_embedding_std_leaves_the_core_seed_untouched(tiny_kwargs):
    """0.02 is the core's own seed, so the default policy is an exact no-op."""
    std = float(_model(tiny_kwargs, horizon_embed_std=0.02).horizon_core.horizon_embedding.std())
    assert std < 0.05


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


def test_the_uncalibrated_heads_are_not_already_at_the_trivial_predictor(tiny_kwargs):
    """The negative control: without the policy the log-variance bias is the generic pass's zero,
    which ``smooth_bound`` maps far from $0$, so the calibrated assertion is not vacuous."""
    model = _model(tiny_kwargs, head_init_calibration=False)
    lo, hi = model.logvar_clamp
    bounded = smooth_bound(model.decoder.logvar_head.bias, lo, hi)

    assert float(bounded.abs().min()) > 0.5


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


def test_the_uncalibrated_prior_is_not_at_unit_scale(tiny_kwargs, raw_inputs):
    """The negative control: Xavier-filled, the head's raw output sits near 0 and the sigmoid bound
    maps it around -1, so the calibrated assertion above is not vacuous."""
    model = _model(tiny_kwargs, head_init_calibration=False).eval()
    with torch.no_grad():
        out = model(*raw_inputs)

    assert float(out["logvar_prior"].abs().mean()) > 0.5


def test_the_prior_calibration_preserves_the_zero_kl_start(tiny_kwargs, raw_inputs):
    """The posterior's log-variance residual is built on the prior's raw pre-bound tensor, so pinning
    that tensor moves prior and posterior together and the KL stays exactly zero."""
    model = _model(tiny_kwargs, head_init_calibration=True).train()
    torch.manual_seed(0)
    out = model(*raw_inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


def test_the_a_head_gain_reaches_the_posterior_fusion(tiny_kwargs):
    weight = _model(tiny_kwargs, a_head_gain=2.0).posterior_head.a_head_norm.weight
    assert torch.equal(weight, torch.full_like(weight, 2.0))


def test_the_default_gain_is_the_plain_unit_norm(tiny_kwargs):
    weight = _model(tiny_kwargs, a_head_gain=1.0).posterior_head.a_head_norm.weight
    assert torch.equal(weight, torch.ones_like(weight))


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
    model = _shipped_flag_model(tiny_kwargs).train()
    torch.manual_seed(0)
    out = model(*raw_inputs)

    assert float(out["kld_per_t"].abs().max()) == 0.0
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
