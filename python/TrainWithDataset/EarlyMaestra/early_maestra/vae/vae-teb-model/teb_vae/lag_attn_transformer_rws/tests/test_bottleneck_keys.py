r"""The three keys that decide how the bottleneck behaves, tested for what each one *does*.

``base_decode``, ``posterior_logvar_mode`` and ``source_dropout`` all exist for one measured
problem, and each is asserted here against the mechanism it was added to remove rather than
against the fact that the key parses.

The measurement they answer, from the $125$-epoch production run
(``output/2026-08-05/metrics_history_rr.csv``):

* ``logvar_prior_floor_frac`` climbs $0.017 \to 0.496$ and is still rising at epoch $125$ -- the
  conditional prior's scale is collapsing onto its clamp, and the ``beta_prior`` anchor only
  slowed it, because that anchor's restoring force saturates at $\beta_p / 2$ per dimension while
  the reconstruction's opposing pressure grows as the decoder sharpens.
* Between epochs $55$ and $120$ the KL is **unchanged** ($1.444 \to 1.442$ nats) and the latent
  displacement is **unchanged** (``delta_mu_rms`` $0.084 \to 0.083$), while the held-out
  predictive gain falls from $+3.25$ to $-0.70$. Same rate, same magnitude, degraded content.

The first two keys attack the collapse, and there are exactly **two** gradient paths driving it,
which is why they are two keys and not one:

1. $D_0$ decodes a *sample* from the prior, so its gradient on $\ell^p$ points down without
   limit -- ``base_decode`` removes that.
2. ``logvar_post = smooth_bound(raw_logvar_prior + delta)`` routes $D_1$'s pressure to sharpen
   $\sigma^q$ onto the *prior's* tensor -- ``posterior_logvar_mode`` severs that, and it is the
   path that **survives** ``base_decode='mean'``.

The third attacks the second measurement: a failure at constant rate is a failure of the content
of $\Delta\mu(h^y, a)$, which a rate lever cannot reach.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import TINY_KWARGS

_TOL = 1e-6


def _build(**overrides) -> SeqVaeLagAttnTrfRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfRws(**{**TINY_KWARGS, **overrides})


def _inputs(model: SeqVaeLagAttnTrfRws, seed: int = 0):
    torch.manual_seed(seed)
    batch, steps = 2, model.sequence_length
    return (
        torch.randn(batch, steps, 43),
        torch.randn(batch, steps, 66),
        torch.randn(batch, steps, model.c_u),
    )


# --------------------------------------------------------------------------------------
# base_decode
# --------------------------------------------------------------------------------------
def test_the_mean_mode_decodes_the_prior_mean_itself():
    """``z^p`` *is* ``mu^p``, bitwise -- not a low-variance sample of it."""
    model = _build(base_decode="mean")
    out = model(*_inputs(model))

    assert torch.equal(out["z_prior"], out["mu_prior"])


def test_the_sample_mode_still_draws_a_sample():
    """The negative control: without it the test above would pass on a model whose prior variance
    had merely collapsed, which is the very condition under investigation."""
    model = _build(base_decode="sample")
    out = model(*_inputs(model))

    assert not torch.equal(out["z_prior"], out["mu_prior"])


@pytest.mark.parametrize(
    "mode, base_is_stable", [("mean", True), ("sample", False)]
)
def test_only_the_mean_mode_makes_the_base_forecast_noise_free(mode, base_is_stable):
    """Two forwards over the *same inputs* under different noise draws. Under ``'mean'`` the base
    forecast must not move at all; the full forecast must move in both modes, or the probe would
    be measuring a model that had stopped sampling anywhere."""
    model = _build(base_decode=mode).train()
    inputs = _inputs(model)

    torch.manual_seed(11)
    first = model(*inputs)
    torch.manual_seed(22)
    second = model(*inputs)

    assert torch.equal(first["mu_base"], second["mu_base"]) is base_is_stable
    assert not torch.equal(first["mu_full"], second["mu_full"])


def test_the_mean_mode_leaves_the_kl_exactly_zero_at_init():
    """What ``'mean'`` costs is the bitwise base-equals-full identity, and what it does **not**
    cost is the zero KL: the divergence is a function of the two distributions, not of the samples
    drawn from them."""
    model = _build(base_decode="mean").train()
    out = model(*_inputs(model))

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["mu_post"], out["mu_prior"])
    # And the identity that genuinely does go: the two branches now differ by the posterior's own
    # noise. Stated as an assertion so the cost is recorded rather than discovered.
    assert not torch.equal(out["mu_base"], out["mu_full"])


def test_an_unknown_base_decode_is_refused():
    """Silently sampling because a config said ``'Mean'`` would be a different experiment under
    the arm's name."""
    with pytest.raises(ValueError, match="base_decode"):
        _build(base_decode="MEAN")


# --------------------------------------------------------------------------------------
# posterior_logvar_mode -- the gradient path, measured
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode, reconstruction_reaches_the_prior", [("residual", True), ("independent", False)]
)
def test_only_the_residual_mode_lets_the_full_branch_reach_the_prior_log_variance(
    mode, reconstruction_reaches_the_prior
):
    r"""The whole point of the key, measured on the gradient rather than argued from the algebra.

    The loss here is the **full branch's reconstruction alone** -- no KL, no prior rate -- so any
    gradient arriving at ``prior_head.logvar_prior_head`` came through the posterior's
    log-variance path and nowhere else. Under ``'residual'`` it does; under ``'independent'`` no
    parameter of that head receives a gradient at all.
    """
    model = _build(posterior_logvar_mode=mode).train()
    out = model(*_inputs(model))

    loss = ((out["mu_full"] - 1.0) ** 2).mean() + out["logvar_full"].mean()
    model.zero_grad()
    loss.backward()

    gradients = [
        float(parameter.grad.abs().max())
        for parameter in model.prior_head.logvar_prior_head.parameters()
        if parameter.grad is not None
    ]
    reached = bool(gradients) and max(gradients) > 0.0
    assert reached is reconstruction_reaches_the_prior


def test_the_kl_still_reaches_the_prior_log_variance_under_the_independent_mode():
    """The path that must survive: severing the reconstruction's route must not orphan the head,
    or the prior's scale would be governed by nothing and DDP would see a dead parameter."""
    model = _build(posterior_logvar_mode="independent").train()
    out = model(*_inputs(model))

    kl = model.kld_tensor(
        mu_prior=out["mu_prior"],
        logvar_prior=out["logvar_prior"],
        mu_post=out["mu_post"],
        logvar_post=out["logvar_post"],
    ).sum()
    model.zero_grad()
    kl.backward()

    assert any(
        parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0
        for parameter in model.prior_head.logvar_prior_head.parameters()
    )


def test_the_independent_mode_keeps_the_exact_zero_kl_under_the_shipped_calibration():
    """``head_init_calibration`` pins the prior's raw log-variance to $\\log(5/3)$ and the
    independent head is seeded at the same constant, so both start at log-variance $0$ and the
    divergence is still exactly zero -- the property the shipped config relies on."""
    model = _build(posterior_logvar_mode="independent", head_init_calibration=True).train()
    out = model(*_inputs(model))

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


def test_without_the_calibration_the_independent_mode_starts_off_the_prior():
    """The cost, recorded rather than left to be discovered: with no constant for the two heads to
    agree on, the init KL is real. This is why the comparison model, which has no such
    calibration, ships ``'residual'``."""
    model = _build(posterior_logvar_mode="independent", head_init_calibration=False).train()
    out = model(*_inputs(model))

    assert float(out["kld_per_t"].abs().max()) > _TOL


@pytest.mark.parametrize("mode", ["residual", "independent"])
def test_exactly_one_log_variance_head_exists_and_every_parameter_is_reached(mode):
    """A head that exists and feeds nothing receives no gradient, which under
    ``find_unused_parameters=False`` hangs the run rather than failing it."""
    model = _build(posterior_logvar_mode=mode).train()
    head = model.posterior_head
    built = [
        name
        for name in ("delta_logvar_head", "logvar_post_head")
        if getattr(head, name) is not None
    ]
    assert len(built) == 1

    out = model(*_inputs(model))
    total = out["mu_full"].sum() + out["logvar_full"].sum() + out["kld_per_t"].sum()
    model.zero_grad()
    total.backward()

    dangling = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert dangling == []


# --------------------------------------------------------------------------------------
# source_dropout
# --------------------------------------------------------------------------------------
def test_the_source_rate_reaches_the_source_pathway_and_nothing_else():
    """With the global rate at zero and the source rate high, the target pathway must be bitwise
    stable across draws while the source pathway is not.

    The posterior deltas are perturbed first: at init they are zero, so ``mu_post`` equals
    ``mu_prior`` whatever the source did, and the ``mu_post`` half of this assertion would pass on
    a model that ignored ``source_dropout`` entirely."""
    model = _build(dropout=0.0, source_dropout=0.5).train()
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)

    inputs = _inputs(model)
    torch.manual_seed(11)
    first = model(*inputs)
    torch.manual_seed(22)
    second = model(*inputs)

    assert torch.equal(first["target_state"], second["target_state"])
    assert torch.equal(first["mu_prior"], second["mu_prior"])
    assert not torch.equal(first["source_state"], second["source_state"])
    assert not torch.equal(first["mu_post"], second["mu_post"])


def test_the_null_source_rate_reproduces_the_pre_key_model_at_every_site():
    """``null`` must mean "unchanged model", and that is not one number.

    The pathway sites -- input adapter and source encoder -- always ran at the global ``dropout``,
    so ``null`` must resolve to it there. The posterior fusion's dropout on the attended source
    summary is a site this key INTRODUCED; before it, ``a`` entered the fusion undropped, so
    ``null`` must resolve to $0$ there. Resolving both to ``dropout`` puts p=0.1 inside the
    posterior of every run that leaves the key unset -- invisible in eval mode, and enough to make
    the ``sweep_source_dropout_*`` arms measure 0.1 -> 0.2 instead of off -> 0.2."""
    unset = _build(dropout=0.25, source_dropout=None)
    assert unset.source_dropout == 0.25
    assert unset.posterior_source_dropout == 0.0
    assert float(unset.posterior_head.a_dropout.p) == 0.0

    explicit = _build(dropout=0.25, source_dropout=0.4)
    assert explicit.source_dropout == 0.4
    assert explicit.posterior_source_dropout == 0.4
    assert float(explicit.posterior_head.a_dropout.p) == 0.4


def test_the_source_rate_never_reaches_the_attention_or_the_decoder():
    """Two places it must never go, each for a stated reason: the lag attention's probabilities
    are returned and consumed by the KL attribution, so dropout there breaks
    $\\sum_\\ell \\widetilde K_{t,\\ell} = K_t$; and the shared decoder is invoked twice per
    forward, so any dropout would draw two masks and put independent noise into ``pred_gap``."""
    model = _build(dropout=0.0, source_dropout=0.5)

    assert float(model.lag_attn.attn_dropout.p) == 0.0
    assert all(
        float(module.p) == 0.0
        for module in model.decoder.modules()
        if isinstance(module, torch.nn.Dropout)
    )
