r"""The full-latent prior head: three outputs, no dead parameters, exact bound identity.

The head exists instead of reusing the sibling's ``PriorHead`` because that one also emits a
``decoder_state`` this architecture must not have: reusing it and discarding the tensor would
leave dead parameters that a distributed run must then be told to tolerate. So the tests here
pin the *absence* as much as the outputs.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import BATCH, SEQ_LEN

_D_MODEL, _D_Z = 32, 8
_CLAMP = (-5.0, 3.0)


def _head() -> FullLatentPriorHead:
    torch.manual_seed(0)
    return FullLatentPriorHead(
        d_model=_D_MODEL, d_z=_D_Z, logvar_clamp=_CLAMP, dropout=0.0, mu_scale=5.0
    )


def _state() -> torch.Tensor:
    return torch.randn(BATCH, SEQ_LEN, _D_MODEL, generator=torch.Generator().manual_seed(1))


def test_the_head_returns_three_latent_shaped_tensors():
    mu, logvar, raw_logvar = _head()(_state())
    for tensor in (mu, logvar, raw_logvar):
        assert tensor.shape == (BATCH, SEQ_LEN, _D_Z)


def test_the_bounded_logvar_is_exactly_the_bound_of_the_raw_one():
    """The posterior residual is applied to the raw value; if this identity drifted, the
    zero-delta posterior would no longer reproduce the prior exactly."""
    _, logvar, raw_logvar = _head()(_state())
    assert torch.equal(logvar, smooth_bound(raw_logvar, *_CLAMP))
    assert (logvar > _CLAMP[0]).all() and (logvar < _CLAMP[1]).all()


def test_the_prior_mean_respects_its_saturation_bound():
    mu, _, _ = _head()(_state())
    assert (mu.abs() <= 5.0).all()


def test_no_parameter_is_dead():
    """Every parameter must reach an output; a dead head is the exact failure reusing the
    sibling's PriorHead would have produced."""
    head = _head()
    mu, logvar, _ = head(_state())
    grads = torch.autograd.grad(
        mu.sum() + logvar.sum(), list(head.parameters()), allow_unused=True
    )
    dead = [
        name
        for (name, _), grad in zip(head.named_parameters(), grads)
        if grad is None
    ]
    assert not dead, f"parameters with no path to an output: {dead}"


def test_there_is_no_decoder_state_pathway():
    head = _head()
    assert not hasattr(head, "decoder_state_head")
    assert not hasattr(head, "dec_input_norm")


def test_a_non_positive_mu_scale_is_rejected():
    with pytest.raises(ValueError, match="mu_scale"):
        FullLatentPriorHead(d_model=_D_MODEL, d_z=_D_Z, mu_scale=0.0)


# =======================================================================================
# The optional clock path
# =======================================================================================
def _clock_head(clock_dim: int = 6) -> FullLatentPriorHead:
    """A head built with the clock path, otherwise identical to :func:`_head`.

    The seed is set but the two heads' shared weights are **not** the same draws: the clock path's
    own norm and projection are constructed first and consume the stream. That is why the
    off-state equality below transplants the shared tensors rather than relying on a seed.
    """
    torch.manual_seed(0)
    return FullLatentPriorHead(
        d_model=_D_MODEL,
        d_z=_D_Z,
        logvar_clamp=_CLAMP,
        dropout=0.0,
        mu_scale=5.0,
        clock_dim=clock_dim,
    )


def _clock(clock_dim: int = 6, seed: int = 2) -> torch.Tensor:
    """A clock tensor, broadcastable against the head's state.

    ``seed`` exists because "a different clock" has to mean a different *pattern*. The clock enters
    through the head's own ``LayerNorm``, so a constant offset or a rescaling of one clock is
    exactly the same tensor by the time the projection sees it -- which is the same fact that made
    a constant availability staircase inert in the prior. Two draws differ where it counts.
    """
    return torch.randn(1, SEQ_LEN, clock_dim, generator=torch.Generator().manual_seed(seed))


def test_a_head_without_a_clock_builds_no_clock_parameter():
    """The off-state, and it is an absence rather than a zero. A projection built and left at zero
    would be a parameter with no gradient path on every two-sided cell in the family, which is the
    ``find_unused_parameters=False`` hazard a distributed run then has to be told to tolerate."""
    head = _head()

    assert head.clock_proj is None and head.clock_norm is None
    assert not any("clock" in name for name in dict(head.named_parameters()))


def test_the_clock_path_is_exactly_zero_at_construction():
    r"""The zero-KL start is what makes every KL number in the records comparable across the
    revision, and it rests on this: a zero projection makes the clock's contribution exactly
    $0$ whatever the clock carries, so the head's output at initialisation is the output it had
    before the path existed.

    Asserted on the constructed head rather than only on the model, because the model's own
    post-initialisation pass re-zeroes it and a test that only read the model could not tell a
    constructor that never zeroed from one whose zero was refilled and restored.
    """
    head = _clock_head()

    assert head.clock_proj is not None
    assert torch.equal(head.clock_proj.weight, torch.zeros_like(head.clock_proj.weight))
    assert head.clock_proj.bias is None, "a bias would be an offset the zero cannot cancel"


def test_a_zero_initialised_clock_reproduces_the_clockless_head_exactly():
    """The off-state pinned behaviourally: at initialisation the two heads are the same function.

    The shared tensors are **transplanted** rather than re-seeded, and that is not a convenience.
    Building the clock path consumes the generator before the two residual heads are drawn, so two
    heads at one seed do not share a single weight -- an equality read off a seed would be
    comparing two different models and would fail for a reason that has nothing to do with the
    clock. Transplanting states the claim exactly: *the same head, plus a zeroed clock path, is the
    same function*.

    The clock fed in is nonzero noise, so what makes the addition inert is the projection's zero
    rather than the clock happening to be small.
    """
    plain, clocked = _head(), _clock_head()
    shared = {
        name: tensor for name, tensor in clocked.state_dict().items() if "clock" not in name
    }
    missing, unexpected = plain.load_state_dict(shared, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    state = _state()
    for left, right in zip(plain(state), clocked(state, _clock())):
        assert torch.equal(left, right)


def test_the_zeroed_clock_path_makes_the_output_independent_of_the_clock():
    """The same claim without a second head, which is what a model-level check can afford.

    A projection that was zero would make the prior invariant to what the clock carries; one that
    was refilled by a later initialisation pass would not. Two independent draws rather than one
    shifted -- the head's own LayerNorm would erase a shift, and the invariance would then hold for
    a reason that has nothing to do with the projection.
    """
    head = _clock_head()
    state = _state()

    first = head(state, _clock())
    second = head(state, _clock(seed=3))

    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_a_nonzero_clock_projection_moves_the_prior():
    """The other direction, so the two invariances above are properties of the **zero** rather than
    of a clock that reaches nothing at all.

    Nothing in the head would raise if the projection were dropped from the forward, or added after
    the two input norms where it could be washed out, or multiplied by a constant zero: every one
    of those would satisfy the invariance tests and none would condition the prior on anything. The
    one comparison that separates them is the same head, with a nonzero projection, under two
    clocks -- so what is asserted is that the clock's *content* reaches the output.
    """
    head = _clock_head()
    assert head.clock_proj is not None
    with torch.no_grad():
        head.clock_proj.weight.normal_(std=0.1)

    state = _state()
    mu_first, _, _ = head(state, _clock())
    mu_second, _, _ = head(state, _clock(seed=3))

    assert (mu_second - mu_first).abs().max().item() > 1e-6


def test_the_clock_reaches_the_head_through_its_own_norm_and_not_the_states():
    """The clock is normalised by a LayerNorm of its **own** width, added to the head input ahead
    of the two existing input norms. Two shapes are pinned because they are what makes the path a
    separate one: a projection sharing the source adapter's map would couple the two pathways'
    gradients, which is the thing this design refused."""
    head = _clock_head(clock_dim=6)

    assert head.clock_norm is not None and head.clock_proj is not None
    assert head.clock_norm.normalized_shape == (6,)
    assert head.clock_proj.in_features == 6
    assert head.clock_proj.out_features == _D_MODEL


@pytest.mark.parametrize(
    "built_with, called_with",
    [(True, False), (False, True)],
    ids=["built with a clock, called without", "built without a clock, called with"],
)
def test_a_head_and_a_call_that_disagree_about_the_clock_are_refused(built_with, called_with):
    """Both directions, and neither is tolerated.

    A head built with a clock and called without one leaves its projection with no gradient path,
    which is the distributed hazard again -- and silently, since the forward would still produce a
    correctly shaped prior. A clock handed to a head that cannot use it is a caller believing in a
    conditioning that is not happening, which is worse: every KL the run reports would be read as
    the clocked model's.
    """
    head = _clock_head() if built_with else _head()
    clock = _clock() if called_with else None

    with pytest.raises(ValueError, match="clock"):
        head(_state(), clock)


def test_a_zero_width_clock_is_refused_rather_than_built():
    """A ``clock_dim`` of 0 would build a projection that can carry nothing, which is the same dead
    parameter as the mismatch above wearing a legal-looking configuration. ``None`` is how a head
    says it conditions on the target state alone."""
    with pytest.raises(ValueError, match="clock_dim"):
        FullLatentPriorHead(d_model=_D_MODEL, d_z=_D_Z, clock_dim=0)


def test_zero_init_clock_is_idempotent_and_harmless_without_a_path():
    """The parents call it from their post-initialisation zeroing block, after a generic pass that
    would otherwise refill the projection. It has to be safe to call on a head that has no clock --
    the two-sided cells build one -- and calling it twice must not be different from once."""
    plain = _head()
    plain.zero_init_clock()  # no path: a no-op rather than an error

    head = _clock_head()
    assert head.clock_proj is not None
    with torch.no_grad():
        head.clock_proj.weight.normal_(std=0.1)
    head.zero_init_clock()
    head.zero_init_clock()

    assert torch.equal(head.clock_proj.weight, torch.zeros_like(head.clock_proj.weight))


def test_perturb_posterior_moves_the_posterior_off_the_prior(
    tiny_kwargs, inputs, perturb_posterior
):
    """The shared perturbation fixture must bite on this model, or every KL assertion in the
    suite is vacuous (at init the posterior equals the prior exactly)."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**tiny_kwargs).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)
    assert (out["mu_post"] - out["mu_prior"]).abs().max().item() > 1e-6
