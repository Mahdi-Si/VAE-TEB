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

The arms that change the module tree (mean-only, scalar lift) and the two chunkings are checked the
same way. Each arrangement is checked on its own batch rather than as a union, because two ranks in
a real step see different batches and a parameter reached on one and not another is exactly the
failure that appears hours into a run rather than at its start.
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


def _mixed_weight() -> torch.Tensor:
    """Some anchors scored and some not, which is what a real batch looks like."""
    weight = torch.ones(TINY_BATCH, TINY_SEQ_LEN)
    weight[0, 10:20] = 0.0
    weight[1, 6:12] = 0.0
    return weight


@pytest.mark.parametrize(
    "make_weight,scored",
    [
        (lambda: None, True),
        (lambda: torch.zeros(TINY_BATCH, TINY_SEQ_LEN), False),
        (_mixed_weight, True),
    ],
    ids=["ordinary", "no_anchor_scored", "mixed_validity"],
)
def test_every_parameter_is_reached_whatever_the_batch_scored(make_weight, scored: bool) -> None:
    """The ordinary batch is the control, without which the other cases could pass by reaching
    nothing anywhere. The batch that scored no anchor is the short-circuit path, the one place a
    rank could drop out of the collective: those anchors cluster around every signal gap, and on a
    large enough run one batch will consist of them.

    Args:
        make_weight: Builds the validity signal, or ``None`` for an all-valid one.
        scored: Whether the batch scores at least one anchor.
    """
    model = build_tiny_model()
    metrics = step(model, weight=make_weight())
    assert (float(metrics["scored_anchors"]) > 0.0) is scored
    assert unreached(model) == []


@pytest.mark.parametrize(
    "overrides",
    [
        # Every source channel cold for the whole record: the gating is a multiplication in the
        # graph, never a branch that omits the head.
        dict(source_warmup_steps=tuple(TINY_SEQ_LEN for _ in range(DECLARED_C_U))),
        # A scale head that exists but is never read would be a starved parameter block.
        dict(mean_only_residual=True),
        # The one arm that puts learned weights in the source encoder.
        dict(source_scalar_lift=True),
        # A detached chunk would leave the proposal head partly unreached.
        dict(anchor_chunk=2),
        dict(lag_chunk=2),
    ],
    ids=["source_unavailable", "mean_only", "scalar_lift", "anchor_chunk", "lag_chunk"],
)
def test_every_parameter_is_reached_on_every_arm_and_chunking(overrides) -> None:
    """Each arrangement where a Python branch omitting a module would be tempting.

    Args:
        overrides: Constructor keywords of the arrangement.
    """
    model = build_tiny_model(**overrides)
    step(model)
    assert unreached(model) == []
