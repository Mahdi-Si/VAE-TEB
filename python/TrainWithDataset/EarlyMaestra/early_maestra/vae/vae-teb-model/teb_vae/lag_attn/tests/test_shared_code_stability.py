r"""A standing guard that the shared network classes keep their current behaviour.

``teb_vae/lag_attn`` and ``teb_vae/lag_attn_rws`` construct the *same* encoder, decoder-core,
posterior and attention classes. The raw-signal model opts into behavioural changes on those
shared classes through default-off constructor flags; every such flag must leave this sibling --
which never passes them -- bitwise unchanged. Nothing else in the suite pins that: the sibling's
own tests assert shapes and invariants, not that its numbers stayed put across an edit to a class
it shares.

This file closes that gap. It builds the sibling at a fixed small geometry with default flags and
pins two things a change to any shared-class default (construction or forward) would move:

* the total parameter count, and
* scalar fingerprints of ``mu_full`` (the encoder -> prior -> decoder path), ``kld_per_t`` and
  ``te_lag_map`` (the attention / KL path).

The posterior is perturbed first, deterministically: the delta heads are zero-initialised, so at
init ``kld_per_t`` and ``te_lag_map`` are identically zero and would fingerprint a constant that
guards nothing. The perturbation makes them a genuine function of the shared forward.

The tolerances are deliberately modest -- loose enough to survive a PyTorch point release, tight
enough that any change to what the shared classes compute lands here.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

# A fixed, structurally faithful tiny geometry: num_heads * d_head == d_model and
# d_z % num_heads == 0. Default flags throughout -- head_structured_latent, causal_norm,
# horizon_film and the extra encoder dilations all stay off, which is exactly the configuration
# whose stability the rws flags must not disturb.
_GEOMETRY: dict = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    warmup_period=2,
    c_y=109,
    c_u=58,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    dropout=0.0,
)

# Captured on the current tree (torch 2.7.1). Any change to a shared-class default moves at least
# one of these.
_EXPECTED_PARAMS = 363222
_EXPECTED_FINGERPRINT = {
    # (sum, sum of absolute values) at double precision.
    "mu_full": (-468.6894950156275, 11887.844673658168),
    "kld_per_t": (716.7452148199081, 716.7452148199081),
    "te_lag_map": (716.7452165931463, 716.7452165931463),
}


def _perturb_posterior(model: SeqVaeLagAttn) -> None:
    """Break the zero-init posterior deterministically so its KL readouts are non-vacuous."""
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _build_and_run():
    """Construct the sibling at the fixed geometry and run one fixed-seed forward."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**_GEOMETRY).eval()
    _perturb_posterior(model)

    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(2, 16, 43, generator=generator)
    y_ph = torch.randn(2, 16, 66, generator=generator)
    u_stream = torch.randn(2, 16, 58, generator=generator)

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(y_st, y_ph, u_stream)
    return model, out


def test_the_shared_sibling_parameter_count_is_unchanged():
    model, _ = _build_and_run()
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == _EXPECTED_PARAMS, (
        f"shared-class parameter count moved: {total} != {_EXPECTED_PARAMS}. A default-off rws "
        f"flag leaked into the sibling, or a shared module's construction changed."
    )


def test_the_shared_sibling_forward_fingerprints_are_unchanged():
    _, out = _build_and_run()
    for key, (expected_sum, expected_abs) in _EXPECTED_FINGERPRINT.items():
        tensor = out[key].double()
        torch.testing.assert_close(
            tensor.sum(),
            torch.tensor(expected_sum, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-3,
            msg=f"{key} sum drifted -- a shared-class forward changed",
        )
        torch.testing.assert_close(
            tensor.abs().sum(),
            torch.tensor(expected_abs, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-3,
            msg=f"{key} magnitude drifted -- a shared-class forward changed",
        )


#: The mechanisms the causal family's revision added, and where each one's off-state is guarded
#: **for this sibling**. Four are absent from this constructor entirely, which is the strongest form
#: the guard takes: a flag that cannot be passed cannot leak, whatever its default becomes.
#:
#: ``alibi_slope_scale`` is the exception and is the reason this is a mapping rather than a list. It
#: is an old keyword of the shared attention, not a new one -- what the revision changed is which
#: value the causal configs ship. So it is guarded by its DEFAULT here: the fingerprints above are
#: measured at $1.0$, and a default moved to the causal family's flat $0.0$ would silently reseed
#: this sibling's lag bias and every number it reports.
_REVISION_KEYWORDS: dict = {
    "lag_kv_source": None,
    "prior_availability_input": None,
    "persistence_residual": None,
    "horizon_weight_halflife_steps": None,
    "alibi_slope_scale": 1.0,
}


def test_no_revision_keyword_reaches_this_sibling_except_by_its_old_default():
    """The construction half of the same guard the fingerprints make behaviourally.

    A parameter count and two fingerprints say the sibling's numbers did not move; this says *why*
    it is structurally hard for them to. Four of the five mechanisms are not keywords of this class
    at all, so no config and no default can reach them here, and the fifth is pinned at the value
    the fingerprints were measured under.

    Both halves matter and neither implies the other: a keyword could be added here with a
    behaviour-preserving default and pass the fingerprints while opening a route for a later
    default change; and the fifth keyword's default could move while the four stayed absent.
    """
    import inspect

    parameters = inspect.signature(SeqVaeLagAttn.__init__).parameters

    for name, expected in _REVISION_KEYWORDS.items():
        if expected is None:
            assert name not in parameters, (
                f"{name} became a keyword of the two-sided model; the causal family's mechanisms "
                f"are gated so that this sibling never takes them"
            )
        else:
            assert parameters[name].default == expected, name
