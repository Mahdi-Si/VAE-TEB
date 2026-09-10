r"""Every parameter reaches the backward on every batch, so a distributed run needs no tolerance.

With unused-parameter handling off -- which is what every model in this family runs with -- a
parameter left **out of the graph** on any rank fails the step. The distinction that matters is not
"received a nonzero gradient": a parameter multiplied by an identically zero mask is still in the
graph and receives a zeros gradient, which is what a distributed run wants. What breaks it is a
Python branch that omits a module, on some ranks and not others, on some batches and not others.

This architecture has three arrangements where such a branch would be tempting, and each is
checked here on the batch that would trigger it:

* a rank whose source is entirely unavailable, so every proposal is gated to zero;
* a rank whose batch scored no anchor at all, which the objective short-circuits;
* a rank whose validity is mixed, which is the ordinary case and the control for the other two.

The check is the union over batches, because a parameter reached on one and not another is exactly
the failure that appears hours into a run rather than at its start.
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    DECLARED_C_U,
    TINY_BATCH,
    TINY_SEQ_LEN,
    build_tiny_model,
    tiny_streams,
)


def step(model, *, weight=None, seed: int = 0) -> Dict[str, torch.Tensor]:
    """One forward, one objective, one backward, and the gradients that came back.

    Args:
        model: The model.
        weight: Validity signal, or ``None`` for an all-valid one.
        seed: Seed for the reparameterisation draw.

    Returns:
        The metric mapping.
    """
    y_st, y_ph, u_stream = tiny_streams()
    torch.manual_seed(seed)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    metrics = model.compute_loss(
        outputs,
        torch.cat([y_st, y_ph], dim=-1),
        weight=torch.ones(TINY_BATCH, TINY_SEQ_LEN) if weight is None else weight,
        beta=1.0,
        beta_prior=0.1,
        likelihood="gaussian_nll",
    )["metrics"]
    metrics["total_loss"].backward()
    return metrics


def unreached(model) -> List[str]:
    """Parameter names that received no gradient at all.

    ``None`` rather than zero is the test: a zeros gradient means the parameter was in the graph,
    which is what a distributed run requires.

    Args:
        model: The model, after a backward.

    Returns:
        The names, sorted.
    """
    return sorted(name for name, p in model.named_parameters() if p.grad is None)


def test_every_parameter_is_reached_on_an_ordinary_batch() -> None:
    """The control: without it the two cases below could pass by reaching nothing anywhere."""
    model = build_tiny_model()
    metrics = step(model)
    assert float(metrics["scored_anchors"]) > 0.0
    assert unreached(model) == []


def test_every_parameter_is_reached_when_the_source_is_entirely_unavailable() -> None:
    """The gating is a multiplication in the graph, never a branch that omits the head."""
    model = build_tiny_model(
        source_warmup_steps=tuple(TINY_SEQ_LEN for _ in range(DECLARED_C_U))
    )
    step(model)
    assert unreached(model) == []


def test_every_parameter_is_reached_when_the_batch_scored_no_anchor() -> None:
    """The short-circuit path, which is the one place a rank could drop out of the collective.

    A batch landing entirely inside a signal gap is not hypothetical: those anchors cluster around
    every gap, and on a large enough run one batch will consist of them.
    """
    model = build_tiny_model()
    metrics = step(model, weight=torch.zeros(TINY_BATCH, TINY_SEQ_LEN))
    assert float(metrics["scored_anchors"]) == 0.0
    assert unreached(model) == []


def test_every_parameter_is_reached_under_mixed_validity() -> None:
    """Some anchors scored and some not, which is what a real batch looks like."""
    weight = torch.ones(TINY_BATCH, TINY_SEQ_LEN)
    weight[0, 10:20] = 0.0
    weight[1, 6:12] = 0.0
    model = build_tiny_model()
    metrics = step(model, weight=weight)
    assert 0.0 < float(metrics["scored_anchors"])
    assert unreached(model) == []


def test_the_union_over_batches_leaves_nothing_unreached() -> None:
    """A parameter reached on one batch and not another fails hours into a run, not at its start.

    Two ranks in a real step see different batches, so the property has to hold for each of them
    separately rather than for their union -- which is what checking each arrangement above and
    then their union together establishes.
    """
    weights = (
        None,
        torch.zeros(TINY_BATCH, TINY_SEQ_LEN),
        torch.ones(TINY_BATCH, TINY_SEQ_LEN),
    )
    for weight in weights:
        model = build_tiny_model()
        step(model, weight=weight)
        assert unreached(model) == [], weight is None


def test_the_mean_only_arm_builds_no_head_it_cannot_reach() -> None:
    """A scale head that exists but is never read is a starved parameter block."""
    model = build_tiny_model(mean_only_residual=True)
    step(model)
    assert unreached(model) == []


def test_the_scalar_lift_arm_reaches_its_per_channel_parameters() -> None:
    """The one arm that puts learned weights in the source encoder."""
    model = build_tiny_model(source_scalar_lift=True)
    step(model)
    assert unreached(model) == []


@pytest.mark.parametrize("grid", [dict(anchor_chunk=2), dict(lag_chunk=2)])
def test_chunking_does_not_drop_a_parameter_from_the_graph(grid) -> None:
    """A detached chunk would leave the proposal head partly unreached.

    Args:
        grid: The chunk sizes to run under.
    """
    model = build_tiny_model(**grid)
    step(model)
    assert unreached(model) == []
