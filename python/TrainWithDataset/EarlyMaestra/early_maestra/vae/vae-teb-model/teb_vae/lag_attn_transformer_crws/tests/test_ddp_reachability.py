r"""Every parameter reaches the graph under both guard states and at the shipped switches.

``find_unused_parameters=False`` is what the shipped DDP strategy claims, and the claim is a
statement about *this* composition rather than about either parent: the decoder head is the raw
grid's width, the availability projections exist only because the warm-up brought them into
existence, and the encoder is the architecture parent's. A parameter starved on any of those three
makes the reducer raise on the first production step and on no development-box run.

The two guard states are both exercised because the guarded one is the only configuration in which
the adapters carry an availability projection at all: an ungated model builds none, so a starvation
introduced by that projection would be invisible in the arm every other suite defaults to.
"""
from __future__ import annotations

from typing import List

import pytest

from .conftest import (
    TINY_KWARGS,
    TINY_STRIDE,
    make_stub_batch,
    tiny_warmup_kwargs,
)


def _starved_parameters(module, batch_idx: int) -> List[str]:
    """Backward one training step and name the trainable parameters left without a gradient."""
    module.zero_grad(set_to_none=True)
    loss, _metrics = module.compute_loss_and_metrics(make_stub_batch(4), batch_idx, "train")
    loss.backward()
    return [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "ungated"])
@pytest.mark.parametrize("beta_prior", [0.0, 0.1], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, guarded, beta_prior
) -> None:
    """What actually licenses ``find_unused_parameters=False``, re-earned on this architecture's
    encoders, on the raw decoder head, and on the availability terms the warm-up brings into
    existence.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step.
    """
    kwargs = (
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
        if guarded
        else dict(TINY_KWARGS, anchor_stride=TINY_STRIDE)
    )
    module = task(
        model_kwargs=kwargs,
        hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior},
    )
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under "
        f"find_unused_parameters=False the reducer raises on exactly these."
    )


@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "ungated"])
def test_under_mse_the_starved_set_is_exactly_the_decoder_logvar_head(
    task, perturb_posterior, guarded
) -> None:
    r"""The mirror image, and the justification for the fallback strategy: with mse the decoder
    log-variance head is trainable and unused. **Exactly** that head and nothing else -- if some
    other parameter starved here, ``find_unused_parameters=True`` would be covering for a second
    defect rather than for a documented configuration choice.

    The head is $R$ wide in both guard states, which is the one width no budget can move: the raw
    block is geometry rather than a gate's survivor count, so both arms starve the same tensor at
    the same size."""
    kwargs = (
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
        if guarded
        else dict(TINY_KWARGS, anchor_stride=TINY_STRIDE)
    )
    module = task(model_kwargs=kwargs, hparams={"likelihood": "mse"})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert set(starved) == {"decoder.logvar_head.weight", "decoder.logvar_head.bias"}, starved
    assert (
        module.orig_model.decoder.logvar_head.bias.numel()
        == module.orig_model.decoder_out_channels
        == module.orig_model.raw_per_step
    )


# =================================================================================================
# The revision's new parameters, and the same claim re-earned on each
#
# Three of the mechanisms this family added build a parameter, and every one of them is exactly the
# shape ``find_unused_parameters=False`` refuses: a tensor whose gradient path is easy to lose and
# whose absence changes no shape. The prior's clock projection is fed by a DETACHED encode, so a
# forward that used the clock and nothing else would starve it; the persistence weight enters the
# mean head alone, so a decode that dropped it would starve it while every forecast stayed correctly
# shaped; and the flat lag bias is seeded to exact zeros, which is the state in which a multiply
# that had been dropped is invisible.
#
# Re-earned rather than argued, on the arm the configs ship, because a starved parameter makes the
# reducer raise on the first production step and on no development-box run.
# =================================================================================================
#: The shipped architecture switches, at the values the causal configs carry.
_SHIPPED_SWITCHES = dict(
    lag_kv_source="conv_stem",
    prior_availability_input=True,
    horizon_weight_halflife_steps=5.0,
    alibi_slope_scale=0.0,
    lag_bias_init="alibi_decay",
)


#: The parameters those switches build, by state-dict name. Written out rather than discovered: a
#: reachability test over whatever a model happened to build would pass on a model that built
#: nothing, which is the state every one of these keys is one config line away from. The
#: persistence weight is absent because this row declines the residual: its target is the raw
#: signal, so there is no stored coefficient to persist.
_EXPECTED_NEW_PARAMETERS = (
    "prior_head.clock_proj.weight",
    "lag_attn.lag_score_bias",
)


@pytest.mark.parametrize("beta_prior", [0.0, 0.1], ids=["unanchored", "anchored"])
def test_no_parameter_the_revision_added_is_left_without_a_gradient(
    task, perturb_posterior, beta_prior
):
    """The same claim as above at the shipped switches, and the parameters they build.

    Asserted in two halves. The starved set must be empty, which is what the reducer checks; and
    the new parameters must be *present*, because an empty starved set is also what a model that
    built none of them would report. Under ``lag_kv_source: conv_stem`` this also covers the source
    pathway the detached prior clock encodes through: it must still reach the graph through the
    matched forward.
    """
    module = task(
        model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, **_SHIPPED_SWITCHES),
        hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior},
    )
    perturb_posterior(module.orig_model)
    names = dict(module.orig_model.named_parameters())
    for expected in _EXPECTED_NEW_PARAMETERS:
        assert expected in names, expected

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under "
        f"find_unused_parameters=False the reducer raises on exactly these."
    )
