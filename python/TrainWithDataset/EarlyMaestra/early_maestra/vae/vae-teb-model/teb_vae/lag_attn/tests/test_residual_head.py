r"""The posterior is a residual around the prior, and that is what makes $K_t \equiv 0$ at init.

The posterior log-variance is built as a zero-initialised delta on the prior's **pre-bound raw**
log-variance, then bounded once. Two things could quietly break that and neither would raise:

* bounding the prior first and adding the delta afterwards -- the bound is a sigmoid, so
  $\mathrm{bound}(\mathrm{bound}(r) + 0) \ne \mathrm{bound}(r)$, and the posterior would start
  a little way off the prior for no reason;
* keeping the independent log-variance head around beside the residual one, leaving a parameter
  that receives no gradient. Under DDP with ``find_unused_parameters=False`` that is not a waste,
  it is a crash.

These test the head's own arithmetic. That the *model* zero-initialises the delta heads is the
model's business and is asserted where the model is.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn.nets.heads import PosteriorHead, PriorHead

_D_MODEL, _D_Z, _NUM_HEADS, _D_HEAD = 32, 8, 4, 8
_BATCH, _SEQ_LEN = 2, 16
_LO, _HI = -5.0, 3.0


def _make_posterior(
    head_structured: bool, posterior_logvar_mode: str = "residual"
) -> PosteriorHead:
    torch.manual_seed(0)
    return PosteriorHead(
        d_model=_D_MODEL,
        d_z=_D_Z,
        dropout=0.0,
        head_structured=head_structured,
        num_heads=_NUM_HEADS,
        d_head=_D_HEAD,
        posterior_logvar_mode=posterior_logvar_mode,
    ).eval()


def _zero_delta_heads(head: PosteriorHead) -> None:
    """Zero both delta heads, reproducing what the model does at construction."""
    for attr in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(head, attr)
        layers = list(module) if isinstance(module, nn.ModuleList) else [module]
        with torch.no_grad():
            for layer in layers:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()


def _posterior_inputs(head_structured: bool):
    generator = torch.Generator().manual_seed(0)
    h_y = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=generator)
    if head_structured:
        a = torch.randn(_BATCH, _SEQ_LEN, _NUM_HEADS, _D_HEAD, generator=generator)
    else:
        a = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL, generator=generator)
    mu_prior = torch.randn(_BATCH, _SEQ_LEN, _D_Z, generator=generator)
    raw_logvar_prior = torch.randn(_BATCH, _SEQ_LEN, _D_Z, generator=generator)
    return h_y, a, mu_prior, raw_logvar_prior


@pytest.mark.parametrize(
    "mode, built, absent",
    [
        ("residual", "delta_logvar_head", "logvar_post_head"),
        ("independent", "logvar_post_head", "delta_logvar_head"),
    ],
)
def test_exactly_one_log_variance_head_is_built(mode, built, absent):
    """The unused head is not disabled, it is not built -- there is nothing to dangle.

    That is a DDP requirement rather than tidiness: a head that exists and feeds nothing receives
    no gradient, and under ``find_unused_parameters=False`` that hangs the run rather than failing
    it. Both attributes are always *present*; exactly one is non-``None``."""
    for head_structured in (False, True):
        head = _make_posterior(head_structured, posterior_logvar_mode=mode)
        assert getattr(head, built) is not None
        assert getattr(head, absent) is None


def test_every_parameter_of_the_built_head_reaches_the_output():
    """The other half of the same requirement, measured rather than asserted from the structure:
    every parameter must receive a gradient from the head's own output."""
    for mode in ("residual", "independent"):
        for head_structured in (False, True):
            head = _make_posterior(head_structured, posterior_logvar_mode=mode)
            mu_post, logvar_post = head(*_posterior_inputs(head_structured))
            head.zero_grad()
            (mu_post.sum() + logvar_post.sum()).backward()
            dangling = [
                name for name, parameter in head.named_parameters() if parameter.grad is None
            ]
            assert dangling == [], f"{mode}/{head_structured}: {dangling}"


def test_retired_flags_are_not_constructor_arguments():
    """Smooth bounding is the model, not an option.

    ``posterior_logvar`` stays refused under its old name: the boolean it used to be is not the
    ``posterior_logvar_mode`` choice that replaced it, and a config carrying the retired spelling
    should fail rather than resolve to a default that happens to look plausible."""
    for retired in ("logvar_bound", "posterior_logvar"):
        with pytest.raises(TypeError):
            PosteriorHead(d_model=_D_MODEL, d_z=_D_Z, **{retired: "whatever"})


@pytest.mark.parametrize("head_structured", [False, True])
def test_zeroed_deltas_reproduce_the_prior_exactly(head_structured):
    head = _make_posterior(head_structured)
    _zero_delta_heads(head)
    h_y, a, mu_prior, raw_logvar_prior = _posterior_inputs(head_structured)

    with torch.no_grad():
        mu_post, logvar_post = head(h_y, a, mu_prior, raw_logvar_prior)

    logvar_prior = smooth_bound(raw_logvar_prior, _LO, _HI)
    # Exact, not approximate: tanh(0) is exactly 0 and the bound is applied once to the same
    # raw value the prior bounded.
    assert torch.equal(mu_post, mu_prior)
    assert (logvar_post - logvar_prior).abs().max().item() < 1e-7


@pytest.mark.parametrize("head_structured", [False, True])
def test_nonzero_deltas_move_the_posterior_off_the_prior(head_structured):
    """The mirror of the test above: it must be possible to fail it."""
    head = _make_posterior(head_structured)
    h_y, a, mu_prior, raw_logvar_prior = _posterior_inputs(head_structured)

    with torch.no_grad():
        mu_post, logvar_post = head(h_y, a, mu_prior, raw_logvar_prior)

    logvar_prior = smooth_bound(raw_logvar_prior, _LO, _HI)
    assert not torch.allclose(mu_post, mu_prior)
    assert not torch.allclose(logvar_post, logvar_prior)


def test_the_residual_is_taken_on_the_raw_prior_not_the_bounded_one():
    """Bounding twice would not be a no-op, and the zero at init would stop being exact."""
    head = _make_posterior(head_structured=False)
    _zero_delta_heads(head)
    h_y, a, mu_prior, raw_logvar_prior = _posterior_inputs(head_structured=False)

    with torch.no_grad():
        _, logvar_post = head(h_y, a, mu_prior, raw_logvar_prior)

    correct = smooth_bound(raw_logvar_prior, _LO, _HI)
    double_bounded = smooth_bound(smooth_bound(raw_logvar_prior, _LO, _HI), _LO, _HI)

    assert torch.allclose(logvar_post, correct, atol=1e-6)
    assert not torch.allclose(logvar_post, double_bounded, atol=1e-3)


def test_the_posterior_requires_the_raw_prior_logvar():
    head = _make_posterior(head_structured=False)
    h_y, a, mu_prior, _ = _posterior_inputs(head_structured=False)
    with pytest.raises(ValueError, match="raw_logvar_prior"):
        head(h_y, a, mu_prior)


def test_posterior_rejects_an_indivisible_latent_under_head_structure():
    with pytest.raises(ValueError, match="d_z % num_heads"):
        PosteriorHead(d_model=_D_MODEL, d_z=9, head_structured=True, num_heads=4, d_head=_D_HEAD)


@pytest.mark.parametrize("scale", ["delta_mu_scale", "delta_logvar_scale"])
def test_posterior_rejects_a_nonpositive_scale(scale):
    with pytest.raises(ValueError, match=scale):
        PosteriorHead(d_model=_D_MODEL, d_z=_D_Z, **{scale: 0.0})


def test_prior_head_returns_the_pre_bound_raw_logvar():
    """The fourth return value is what makes the exact residual possible."""
    torch.manual_seed(0)
    head = PriorHead(d_model=_D_MODEL, d_z=_D_Z, dropout=0.0).eval()
    h_y = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL)

    with torch.no_grad():
        mu, logvar, decoder_state, raw = head(h_y)

    assert raw.shape == logvar.shape == mu.shape == (_BATCH, _SEQ_LEN, _D_Z)
    assert decoder_state.shape == (_BATCH, _SEQ_LEN, _D_MODEL)
    # The reported logvar is exactly the smooth bound of the returned raw value.
    assert torch.allclose(logvar, smooth_bound(raw, _LO, _HI), atol=1e-6)
    assert torch.all(logvar > _LO) and torch.all(logvar < _HI)


def test_prior_mean_is_bounded_by_mu_scale():
    torch.manual_seed(0)
    head = PriorHead(d_model=_D_MODEL, d_z=_D_Z, dropout=0.0, mu_scale=2.0).eval()
    # Drive the head hard enough to saturate the tanh.
    h_y = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL) * 50.0
    with torch.no_grad():
        mu, _, _, _ = head(h_y)
    assert mu.abs().max().item() <= 2.0


def test_prior_rejects_a_nonpositive_mu_scale():
    with pytest.raises(ValueError, match="mu_scale"):
        PriorHead(d_model=_D_MODEL, d_z=_D_Z, mu_scale=0.0)
